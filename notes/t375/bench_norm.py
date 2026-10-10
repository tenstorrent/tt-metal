# t375: micro-benchmark of the conv VAE decoder's fused RMSNorm+SiLU (dit_rms_norm_unary_fused) on the full 4x8
# mesh, per-chip decoder shapes at 1088x1920/145f, row-major in/out like the decoder. Prints per variant and shape:
# min wall ms over REPS (sync each), PCC / max abs diff vs the fp32 torch reference and vs the current config (V0).
import os, sys, time
import torch
import ttnn

SHAPES = {  # name: (per-chip shape, count in the decoder)
    "res128": ((1, 145, 68, 60, 128), 9),
    "res256": ((1, 145, 34, 30, 256), 12),
    "res512b": ((1, 73, 34, 30, 512), 8),
    "res512a": ((1, 37, 17, 15, 512), 4),
    "res1024": ((1, 19, 9, 8, 1024), 4),
}
REPS = 5


def ckc(mesh, fid, fp32, approx=False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=fid, math_approx_mode=approx, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
    F = ttnn.MathFidelity
    S = ttnn.UnaryOpType.SILU
    variants = {
        "V0_hifi4_fp32_silu": (ckc(mesh, F.HiFi4, True), S, ttnn.ROW_MAJOR_LAYOUT),
        "V1_hifi4_fp32_noact": (ckc(mesh, F.HiFi4, True), None, ttnn.ROW_MAJOR_LAYOUT),
        "V2_hifi2_fp32_silu": (ckc(mesh, F.HiFi2, True), S, ttnn.ROW_MAJOR_LAYOUT),
        "V3_lofi_fp32_silu": (ckc(mesh, F.LoFi, True), S, ttnn.ROW_MAJOR_LAYOUT),
        "V4_hifi4_bf16_silu": (ckc(mesh, F.HiFi4, False), S, ttnn.ROW_MAJOR_LAYOUT),
        "V5_lofi_bf16_silu": (ckc(mesh, F.LoFi, False), S, ttnn.ROW_MAJOR_LAYOUT),
        "V6_hifi4_fp32_silu_TILEin": (ckc(mesh, F.HiFi4, True), S, ttnn.TILE_LAYOUT),
        "V7_hifi4_bf16_noact": (ckc(mesh, F.HiFi4, False), None, ttnn.ROW_MAJOR_LAYOUT),
    }
    only = os.environ.get("BENCH_SHAPES")
    tot = {v: 0.0 for v in variants}
    for name, (shape, count) in SHAPES.items():
        if only and name not in only.split(","):
            continue
        torch.manual_seed(0)
        # Decoder activations are not zero-mean; give each channel its own offset and scale.
        C = shape[-1]
        x = torch.randn(shape) * (0.5 + torch.rand(C)) + 0.3 * torch.randn(C)
        x = x.to(torch.bfloat16)
        xf = x.float()
        ref_norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-8)
        refs = {S: torch.nn.functional.silu(ref_norm), None: ref_norm}
        outs = {}
        tin = {
            lay: ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=lay, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )
            for lay in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)
        }
        for vname, (cfg, act, lay) in variants.items():
            kw = dict(epsilon=1e-8, compute_kernel_config=cfg)
            if act is not None:
                kw["activation"] = act
            y = None
            for _ in range(2):
                y = ttnn.experimental.dit_rms_norm_unary_fused(tin[lay], **kw)
                ttnn.synchronize_device(mesh)
                ttnn.deallocate(y)
            ts = []
            for _ in range(REPS):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                y = ttnn.experimental.dit_rms_norm_unary_fused(tin[lay], **kw)
                ttnn.synchronize_device(mesh)
                ts.append(time.perf_counter() - t0)
                if _ < REPS - 1:
                    ttnn.deallocate(y)
            out = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(shape)
            ttnn.deallocate(y)
            ref = refs[act]
            pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
            mad = (out - ref).abs().max().item()
            line = f"BENCH {name} {vname} min_ms={min(ts)*1e3:.3f} med_ms={sorted(ts)[REPS//2]*1e3:.3f} pcc_ref={pcc:.7f} maxabs_ref={mad:.4g}"
            if act is S:
                if vname.startswith("V0"):
                    outs["V0"] = out
                else:
                    d = out - outs["V0"]
                    p0 = torch.corrcoef(torch.stack([out.flatten(), outs["V0"].flatten()]))[0, 1].item()
                    line += f" pcc_v0={p0:.7f} maxabs_v0={d.abs().max().item():.4g} bitident_v0={bool((d == 0).all())}"
            print(line, flush=True)
            tot[vname] += min(ts) * 1e3 * count
        for t in tin.values():
            ttnn.deallocate(t)
    for v, ms in tot.items():
        print(f"BENCH_TOTAL {v} weighted_ms={ms:.2f}", flush=True)
    ttnn.close_mesh_device(mesh)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
