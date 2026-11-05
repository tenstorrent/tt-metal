# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: TT DiT driven by native CUDA conditioning.

mode=free     : independent 50-step TT denoising from seeded noise, CUDA conditioning.
mode=teacher  : every step consumes the CUDA input latent; measures local per-step error.
Neither mode is a qualification run (conditioning is activation-injected).
"""
import argparse, json, time
from pathlib import Path

import torch


def metrics(actual, reference):
    a, b = actual.double().flatten(), reference.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    return {
        "pcc": float((ac @ bc) / (ac.norm() * bc.norm()).clamp_min(1e-30)),
        "relative_rms": float((a - b).norm() / b.norm().clamp_min(1e-30)),
        "max_abs": float((a - b).abs().max()),
        "gain": float((a @ b) / (b @ b).clamp_min(1e-30)),
    }


def install_manual_attention(t, group):
    """Diagnostic SDPA replacement: FP32 QK^T scores and softmax, BF16 probabilities, FP32-accumulated PV."""
    masks = {}
    native = t.transformer.scaled_dot_product_attention

    def sdpa(
        q,
        k,
        v,
        *,
        attn_mask=None,
        is_causal=False,
        scale=None,
        compute_kernel_config=None,
        memory_config=None,
        program_config=None,
        **kwargs,
    ):
        if is_causal or attn_mask is None or kwargs:
            # Text-encoder causal attention keeps the qualified native kernel.
            return native(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
                compute_kernel_config=compute_kernel_config,
                memory_config=memory_config,
                program_config=program_config,
                **kwargs,
            )
        key = id(attn_mask)
        if key not in masks:
            masks.clear()
            masks[key] = (attn_mask, t.typecast(attn_mask, t.float32))
        mask = masks[key][1]
        heads = q.shape[1]
        outs = []
        for start in range(0, heads, group):
            end = min(heads, start + group)
            qh, kh, vh = q[:, start:end], k[:, start:end], v[:, start:end]
            s = t.matmul(
                qh,
                t.transpose(kh, -2, -1),
                dtype=t.float32,
                compute_kernel_config=compute_kernel_config,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            s2 = t.add(t.multiply(s, scale), mask)
            s.deallocate()
            pr = t.softmax(s2, dim=-1, numeric_stable=True, compute_kernel_config=compute_kernel_config)
            s2.deallocate()
            pb = t.typecast(pr, t.bfloat16)
            pr.deallocate()
            o = t.matmul(
                pb, vh, dtype=t.float32, compute_kernel_config=compute_kernel_config, memory_config=t.DRAM_MEMORY_CONFIG
            )
            pb.deallocate()
            outs.append(t.typecast(o, t.bfloat16))
            o.deallocate()
        return outs[0] if len(outs) == 1 else t.concat(outs, dim=1, memory_config=memory_config)

    t.transformer.scaled_dot_product_attention = sdpa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["free", "teacher"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--teacher-steps", type=int, nargs="*")
    p.add_argument("--attention", choices=["native", "manual"], default="native")
    p.add_argument("--head-group", type=int, default=6)
    args = p.parse_args()
    import ttnn as t

    if args.attention == "manual":
        install_manual_attention(t, args.head_group)
    from models.experimental.moss_image.tt.dit import TTMossDiT
    from models.experimental.moss_image.tt.scheduler import guided_euler_step
    from models.experimental.moss_image.tt.vendor.pipeline_moss_image import MOSSImageScheduler

    ref = Path(args.reference)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "attention": args.attention,
        "mode": args.mode,
        "reference": str(ref),
        "activation_injection": "CUDA conditioning"
        + (" and per-step CUDA input latents" if args.mode == "teacher" else ""),
        "steps": {},
    }

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    mesh = t.open_mesh_device(
        mesh_shape=t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=131072, trace_region_size=0
    )
    try:
        model = TTMossDiT(Path(args.checkpoint), mesh)
        contexts = {}
        for branch in ("positive", "negative"):
            c = torch.load(ref / "conditioning" / f"{branch}.pt", weights_only=True)
            h = c["hidden_states"][:, c["attention_mask"][0].bool()].contiguous()
            contexts[branch] = model.upload(h)
            report[f"{branch}_tokens"] = int(h.shape[1])
        height = width = 64
        scheduler = MOSSImageScheduler.from_pretrained(Path(args.checkpoint) / "scheduler", local_files_only=True)
        scheduler.set_timesteps(args.steps, device="cpu")
        noise = torch.randn(
            (1, 128, height, width), generator=torch.Generator("cpu").manual_seed(42), dtype=torch.float32
        )
        report["noise_bitwise_equal"] = bool(
            torch.equal(noise, torch.load(ref / "steps/000/input_latents.pt", weights_only=True))
        )
        latent = model.upload(noise.flatten(2).transpose(1, 2), dtype=t.float32)
        indices = range(len(scheduler.timesteps))
        if args.mode == "teacher" and args.teacher_steps:
            indices = args.teacher_steps
        for index in indices:
            timestep = scheduler.timesteps[index]
            d = ref / f"steps/{index:03d}"
            stamp = time.perf_counter()
            if args.mode == "teacher":
                src = d / "input_latents.pt" if index == 0 else ref / f"steps/{index-1:03d}/updated_latents.pt"
                x = torch.load(src, weights_only=True)
                latent = model.upload(x.flatten(2).transpose(1, 2), dtype=t.float32)
            pos = model(latent, float(timestep), contexts["positive"], height, width)
            neg = model(latent, float(timestep), contexts["negative"], height, width, skip_layers=(22,))
            dt = float(scheduler.sigmas[index + 1] - scheduler.sigmas[index])
            latent, guided = guided_euler_step(latent, pos, neg, 3.0, dt)
            to = lambda v: model.host(v).transpose(1, 2).reshape(1, 128, height, width).float()
            entry = {"seconds": time.perf_counter() - stamp}
            updated = torch.load(d / "updated_latents.pt", weights_only=True)
            entry["latent"] = metrics(to(latent), updated)
            if args.mode == "teacher":
                # Euler step increment: isolates the local guided velocity error at every step.
                entry["increment"] = metrics(to(latent) - x, updated - x)
            if (d / "positive_prediction.pt").is_file():
                entry["positive"] = metrics(
                    to(pos), torch.load(d / "positive_prediction.pt", weights_only=True).float()
                )
                entry["negative"] = metrics(
                    to(neg), torch.load(d / "negative_prediction.pt", weights_only=True).float()
                )
                entry["guided"] = metrics(to(guided), torch.load(d / "guided_prediction.pt", weights_only=True).float())
            report["steps"][f"{index:03d}"] = entry
            save()
            print(
                index,
                json.dumps({k: round(v["relative_rms"] * 100, 4) for k, v in entry.items() if isinstance(v, dict)}),
                flush=True,
            )
            del pos, neg, guided
        if args.mode == "free":
            torch.save(model.host(latent), out / "final_latent_tokens.pt")
        report["status"] = "complete"
        save()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        save()
        raise
    finally:
        t.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
