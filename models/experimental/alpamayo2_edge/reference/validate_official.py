# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One-time validation of the pure-torch reference against the official Alpamayo 2 Edge stack.

`dump` runs in a separate venv with the official stack (NVlabs `alpamayo2_edge` on transformers 5.17, see
README.md). It feeds `synthetic_sample(0)` through the official helper and `sample_trajectories_from_data` in fp32 /
SDPA and saves the inputs and the intermediates the TT tests take from the reference. `compare` runs in the repo
venv: it rebuilds the inputs with `prepare_sample_inputs`, which must match bitwise, then runs `ReferenceModel` on
the official CoT and noise and checks every tensor by PCC.

    dump:    <official venv>/bin/python models/experimental/alpamayo2_edge/reference/validate_official.py dump --out DIR
    compare: python models/experimental/alpamayo2_edge/reference/validate_official.py compare --dump DIR
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
# Run as a script, Python puts reference/ first on sys.path, where config.py, inputs.py, ... would shadow
# top-level modules of the same name; the package is imported from the repo root instead.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path[0] = str(REPO_ROOT)
elif str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from models.experimental.alpamayo2_edge.reference.inputs import synthetic_sample

SAMPLE_SEED = 0
# Seeds torch's global RNG (CoT sampling) and the diffusion generator, as `alpamayo2_edge.inference.predict` does.
OFFICIAL_SEED = 0
PCC_MIN = 0.99999
# PCC is blind to a global scale or offset; measured max_rel is <= 1.1e-5.
MAX_REL = 1e-3
MISSING_ANCHOR = "No <traj_future_start> token found"
CASES = ("sampled", "long_forced")
# Two vision checks, then per case: prefill logits, 2 CoT logits, K/V of 2 layers, 2 velocities, actions, xyz, rot.
EXPECTED_CHECKS = 2 + 12 * len(CASES)
DUMP_FILE = "official.pt"


def log(message: str) -> None:
    print(f"[validate_official {time.strftime('%H:%M:%S')}] {message}", flush=True)


class Heartbeat:
    """Prints while a long CPU stage runs, since the job broker kills jobs that stay silent for 300 s."""

    def __init__(self, period_s: float = 30.0):
        self.period_s = period_s
        self.stage = "start"
        self.started = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.period_s):
            log(f"... {self.stage} ({time.time() - self.started:.0f} s elapsed)")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


# ---------------------------------------------------------------------------------------------------------------------
# dump (official venv)
# ---------------------------------------------------------------------------------------------------------------------


def _official_versions() -> dict:
    from importlib.metadata import distribution

    import transformers

    direct_url = json.loads(distribution("alpamayo2-edge").read_text("direct_url.json") or "{}")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "alpamayo2_edge_commit": direct_url.get("vcs_info", {}).get("commit_id"),
    }


def _run_official(model, prepared: dict, forced: torch.Tensor | None, kv_layers: tuple[int, ...]) -> dict:
    """`sample_trajectories_from_data` through `inference.predict`, recording intermediates with hooks.

    With `forced`, a logits processor leaves only the forced token at each step, so sampling reproduces it.
    """
    from transformers.generation.logits_process import LogitsProcessor

    from alpamayo2_edge.inference import predict

    rec: dict = {"lm_head": [], "steps": []}
    prompt_len = prepared["tokenized_data"]["input_ids"].shape[1]
    vlm_model = model.vlm.model

    class ForceTokens(LogitsProcessor):
        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - prompt_len
            if step < forced.shape[0]:
                keep = scores[:, forced[step]].clone()
                scores.fill_(-torch.inf)
                scores[:, forced[step]] = keep
            return scores

    def on_lm_head(_module, _args, output):
        rec["lm_head"].append(output[:, -1].detach().float().clone())
        if len(rec["lm_head"]) % 16 == 0:
            log(f"decode step {len(rec['lm_head'])}")

    hooks = [
        vlm_model.visual.register_forward_hook(
            lambda _m, _a, out: rec.__setitem__("vision_encoder", out.last_hidden_state.detach().clone())
        ),
        vlm_model.projector.register_forward_hook(lambda _m, _a, out: rec.__setitem__("vision", out.detach().clone())),
        model.vlm.get_output_embeddings().register_forward_hook(on_lm_head),
    ]

    original_generate = model.vlm.generate

    def generate(*args, **kwargs):
        log("prefill done; decoding")
        if forced is not None:
            kwargs["logits_processor"].append(ForceTokens())
        out = original_generate(*args, **kwargs)
        rec["sequences"] = out.sequences.detach().clone()
        return out

    original_sample_actions = model._sample_actions

    def sample_actions(cache, prefix_length, position_ids, attention_mask, *args, **kwargs):
        log(f"CoT cached ({prefix_length} rows); running the expert")
        rec["prefix_length"] = prefix_length
        rec["kv"] = {
            layer: (cache.layers[layer].keys.detach().clone(), cache.layers[layer].values.detach().clone())
            for layer in kv_layers
        }
        rec["expert_position_ids"] = position_ids.detach().clone()
        rec["expert_attention_mask"] = attention_mask.detach().clone()
        out = original_sample_actions(cache, prefix_length, position_ids, attention_mask, *args, **kwargs)
        rec["actions"] = out.detach().clone()
        return out

    original_diffusion_sample = model.expert.diffusion.sample

    def diffusion_sample(*args, step_fn, **kwargs):
        def recording_step_fn(x, t):
            v = step_fn(x=x, t=t)
            rec["steps"].append((x.detach().clone(), t.detach().clone(), v.detach().clone()))
            return v

        return original_diffusion_sample(*args, step_fn=recording_step_fn, **kwargs)

    model.vlm.generate = generate
    model._sample_actions = sample_actions
    model.expert.diffusion.sample = diffusion_sample
    settings = dict(
        seed=OFFICIAL_SEED,
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_sets=1,
        num_traj_samples=1,
        max_generation_length=256,
        diffusion_kwargs={"inference_step": 10},
    )
    try:
        xyz, rot, _ = predict(model, prepared, settings, "cpu")
        rec.update(xyz=xyz.detach().clone(), rot=rot.detach().clone(), error=None)
    except ValueError as exc:
        # A sampled CoT without the anchor fails after generation; the sequences are already recorded.
        if MISSING_ANCHOR not in str(exc) or "sequences" not in rec:
            raise
        rec["error"] = str(exc)
    finally:
        for hook in hooks:
            hook.remove()
        del model.vlm.generate, model._sample_actions, model.expert.diffusion.sample
    rec["settings"] = settings
    return rec


def dump(out_dir: Path) -> None:
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs
    from alpamayo2_edge.models.utils import fuse_traj_tokens

    from models.experimental.alpamayo2_edge.reference.config import load_config, resolve_model_dir
    from models.experimental.alpamayo2_edge.reference.forced_cot import DRIVING_COT_TEXT_TOKENS, driving_text_cot
    from models.experimental.alpamayo2_edge.reference.model import make_forced_cot

    out_dir.mkdir(parents=True, exist_ok=True)
    with Heartbeat() as beat:
        versions = _official_versions()
        log(f"official stack: {versions}")
        model_dir = resolve_model_dir()
        beat.stage = "loading the official model"
        log(f"{beat.stage} from {model_dir}")
        model, _ = load_model(
            str(model_dir), device="cpu", dtype="float32", attn_implementation="sdpa", local_files_only=True
        )
        cfg = model.config
        anchor = cfg.traj_ids["future_start"]
        kv_layers = (0, cfg.vlm_config.text_config.num_hidden_layers - 1)

        beat.stage = "preparing inputs"
        log(beat.stage)
        sample = synthetic_sample(SAMPLE_SEED)
        prepared = prepare_model_inputs(sample, cfg, model.tokenizer)
        tok = prepared["tokenized_data"]
        input_ids = fuse_traj_tokens(
            model.history_traj_tokenizer,
            model.future_traj_tokenizer,
            tok["input_ids"],
            {k: prepared[k] for k in ("ego_history_xyz", "ego_history_rot")},
            cfg.traj_ids,
        )
        position_ids, rope_deltas = model.vlm.model.get_rope_index(
            input_ids,
            mm_token_type_ids=tok["mm_token_type_ids"],
            image_grid_thw=tok["image_grid_thw"],
            attention_mask=tok["attention_mask"],
        )
        inputs = {
            "input_ids": input_ids,
            "attention_mask": tok["attention_mask"],
            "mm_token_type_ids": tok["mm_token_type_ids"],
            "pixel_values": tok["pixel_values"],
            "image_grid_thw": tok["image_grid_thw"],
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
            "ego_history_xyz": prepared["ego_history_xyz"],
            "ego_history_rot": prepared["ego_history_rot"],
        }
        log(f"S = {input_ids.shape[1]}, N = {tok['pixel_values'].shape[0]} patches")

        prompt_len = input_ids.shape[1]
        cases = {}

        def run_case(name: str, forced: torch.Tensor | None) -> dict:
            beat.stage = f"official run: {name}"
            log(beat.stage)
            start = time.time()
            run = _run_official(model, prepared, forced, kv_layers)
            log(f"{name} took {time.time() - start:.0f} s, error: {run['error']}")
            if forced is not None:
                if run["error"] is not None:
                    raise RuntimeError(f"official forced run failed: {run['error']}")
                if not torch.equal(run["sequences"][0, prompt_len : prompt_len + forced.shape[0]], forced):
                    raise RuntimeError("the forced official run did not reproduce the forced CoT")
            return run

        sampled = run_case("sampled CoT", None)
        sampled_cot = sampled["sequences"][0, prompt_len:]
        if sampled["error"] is None:
            cases["sampled"] = (sampled, make_forced_cot(sampled_cot, anchor), "official sampled CoT up to its anchor")
        else:
            forced = make_forced_cot(sampled_cot[:DRIVING_COT_TEXT_TOKENS], anchor)
            source = f"official sampled CoT had no anchor; its first {DRIVING_COT_TEXT_TOKENS} tokens + anchor"
            cases["sampled"] = (run_case("sampled CoT, truncated + anchor", forced), forced, source)

        forced = driving_text_cot(load_config(model_dir))
        source = f"teacher-forced driving_text_cot: {forced.shape[0] - 1} text tokens + anchor"
        cases["long_forced"] = (run_case("long forced CoT", forced), forced, source)

        result = {
            "versions": versions,
            "snapshot": model_dir.name,
            "sample_seed": SAMPLE_SEED,
            "settings": sampled["settings"],
            "sampled_sequence": sampled["sequences"][0].clone(),
            "inputs": inputs,
            "vision_encoder": sampled["vision_encoder"],
            "vision": sampled["vision"],
            "cases": {},
        }
        for name, (run, forced, source) in cases.items():
            log(f"case {name}: {forced.shape[0]} CoT tokens ({source}); prefix length {run['prefix_length']}")
            log(f"case {name} CoT text: " + repr(model.tokenizer.decode(forced, skip_special_tokens=False)))
            result["cases"][name] = {
                "cot_source": source,
                "forced_cot": forced,
                "step_logits": torch.cat(run["lm_head"], dim=0),
                "prefix_length": run["prefix_length"],
                "kv": run["kv"],
                "expert_position_ids": run["expert_position_ids"],
                "expert_attention_mask": run["expert_attention_mask"],
                "expert_steps": run["steps"],
                "actions": run["actions"],
                "xyz": run["xyz"],
                "rot": run["rot"],
            }
        tmp = out_dir / f"{DUMP_FILE}.tmp"
        torch.save(result, tmp)
        tmp.replace(out_dir / DUMP_FILE)
        log(f"wrote {out_dir / DUMP_FILE}")


# ---------------------------------------------------------------------------------------------------------------------
# compare (repo venv)
# ---------------------------------------------------------------------------------------------------------------------


def tensor_stats(ours: torch.Tensor, official: torch.Tensor) -> dict:
    """PCC, max-abs error and max-abs error relative to the official tensor's max magnitude."""
    if ours.shape != official.shape:
        raise ValueError(f"shape mismatch: ours {tuple(ours.shape)} vs official {tuple(official.shape)}")
    a, b = ours.detach().double().flatten(), official.detach().double().flatten()
    max_abs = float((a - b).abs().max())
    scale = float(b.abs().max())
    pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if a.numel() > 1 else float(a == b)
    return {"pcc": pcc, "max_abs": max_abs, "max_rel": max_abs / scale if scale else max_abs, "ref_max": scale}


def check_case(name, case, model, inp, prompt_cache, check, failures, beat) -> None:
    """Teacher-force the case's CoT on a copy of the prefill cache, then run the expert on the official noise."""
    from models.experimental.alpamayo2_edge.reference.action import action_to_traj
    from models.experimental.alpamayo2_edge.reference.text import KVCache

    cache = KVCache(len(prompt_cache.k))
    # decode and the expert extend and crop by rebinding, so the prefill tensors are shared, never modified.
    cache.k, cache.v = list(prompt_cache.k), list(prompt_cache.v)
    forced = case["forced_cot"]
    beat.stage = f"[{name}] reference forced decode of {forced.shape[0]} tokens"
    log(beat.stage)
    cot = model.decode(inp, cache, forced=forced[None])
    if not torch.equal(cot.tokens[0], forced):
        failures.append(f"[{name}] reference decode did not reproduce the forced CoT")
    if cache.seq_len != case["prefix_length"] or int(cot.offsets[0]) != case["prefix_length"]:
        failures.append(
            f"[{name}] CoT prefix: reference cache {cache.seq_len}, offset {int(cot.offsets[0])}, "
            f"official {case['prefix_length']}"
        )
    log(f"[{name}] CoT prefix length: reference {cache.seq_len}, official {case['prefix_length']}")
    steps = forced.shape[0]
    official_logits = case["step_logits"][:steps]
    check(f"[{name}] forced-CoT logits ({steps} steps)", cot.logits[0], official_logits)
    worst = min(range(steps), key=lambda i: tensor_stats(cot.logits[0, i], official_logits[i])["pcc"])
    check(f"[{name}] forced-CoT logits, worst step {worst}", cot.logits[0, worst], official_logits[worst])
    last = model.cfg.text.layers - 1
    if set(case["kv"]) != {0, last}:
        failures.append(f"[{name}] dump has K/V of layers {sorted(case['kv'])}, expected [0, {last}]")
    for layer, (k, v) in case["kv"].items():
        check(f"[{name}] K layer {layer} ({k.shape[2]} rows)", cache.k[layer], k)
        check(f"[{name}] V layer {layer} ({v.shape[2]} rows)", cache.v[layer], v)

    beat.stage = f"[{name}] reference expert"
    log(beat.stage)
    x0, t0, v0 = case["expert_steps"][0]
    check(f"[{name}] expert velocity, step 0 (noise)", model.expert_velocity(cache, cot, inp, x0, t0), v0)
    pccs = [tensor_stats(model.expert_velocity(cache, cot, inp, x, t), v)["pcc"] for x, t, v in case["expert_steps"]]
    worst = min(range(len(pccs)), key=pccs.__getitem__)
    x, t, v = case["expert_steps"][worst]
    check(f"[{name}] expert velocity, worst step {worst}", model.expert_velocity(cache, cot, inp, x, t), v)
    actions = model.sample_actions(cache, cot, inp, x0)
    check(f"[{name}] actions (10 Euler steps)", actions, case["actions"])
    xyz, rot = action_to_traj(actions, inp.ego_history_xyz, inp.ego_history_rot, model.cfg.action)
    official_xyz = case["xyz"].reshape(xyz.shape)
    check(f"[{name}] xyz", xyz, official_xyz)
    check(f"[{name}] rot", rot, case["rot"].reshape(rot.shape))
    log(f"[{name}] max xyz error {float((xyz - official_xyz).abs().max()):.3e} m")


def compare(dump_dir: Path) -> None:
    from models.experimental.alpamayo2_edge.reference.config import load_config, num_layers_override, resolve_model_dir
    from models.experimental.alpamayo2_edge.reference.forced_cot import driving_text_cot
    from models.experimental.alpamayo2_edge.reference.inputs import prepare_sample_inputs
    from models.experimental.alpamayo2_edge.reference.model import ReferenceModel

    if num_layers_override() is not None:
        raise ValueError("the official dump is full depth; unset ALPAMAYO_NUM_LAYERS")
    off = torch.load(dump_dir / DUMP_FILE, weights_only=False)
    log(f"official dump: {off['versions']}, snapshot {off['snapshot']}")
    failures = []
    exact, rows = {}, []

    with Heartbeat() as beat:
        cfg = load_config(resolve_model_dir())
        if cfg.model_dir.name != off["snapshot"]:
            raise ValueError(f"snapshot {cfg.model_dir.name} differs from the dump's {off['snapshot']}")
        beat.stage = "preparing inputs"
        log(beat.stage)
        inp = prepare_sample_inputs(cfg, synthetic_sample(off["sample_seed"]))
        o = off["inputs"]
        pairs = {
            "input_ids": (inp.input_ids, o["input_ids"]),
            "attention_mask": (inp.attention_mask, o["attention_mask"]),
            "mm_token_type_ids": (inp.mm_token_type_ids, o["mm_token_type_ids"]),
            "patches vs pixel_values": (inp.patches, o["pixel_values"]),
            "image_grid_thw": (inp.image_grid_thw, o["image_grid_thw"]),
            "position_ids": (inp.position_ids, o["position_ids"]),
            "rope_deltas": (inp.rope_deltas, o["rope_deltas"]),
            "ego_history_xyz": (inp.ego_history_xyz, o["ego_history_xyz"].reshape(inp.ego_history_xyz.shape)),
            "ego_history_rot": (inp.ego_history_rot, o["ego_history_rot"].reshape(inp.ego_history_rot.shape)),
        }
        for name, (ours, official) in pairs.items():
            same = ours.shape == official.shape and ours.dtype == official.dtype and torch.equal(ours, official)
            exact[name] = same
            detail = f"{tuple(ours.shape)} {ours.dtype}"
            if not same:
                failures.append(f"input {name} is not bitwise equal")
                if ours.shape == official.shape:
                    detail += f", {int((ours != official).sum())} elements differ"
                else:
                    detail += f" vs official {tuple(official.shape)} {official.dtype}"
            log(f"input {name}: {'bitwise equal' if same else 'DIFFERS'} ({detail})")
        log(f"S = {inp.input_ids.shape[1]}, N = {inp.patches.shape[0]}")

        def check(name: str, ours: torch.Tensor, official: torch.Tensor) -> None:
            stats = tensor_stats(ours, official)
            ok = stats["pcc"] >= PCC_MIN and stats["max_rel"] <= MAX_REL
            rows.append({"name": name, "shape": list(ours.shape), **stats, "pass": ok})
            if stats["pcc"] < PCC_MIN:
                failures.append(f"{name}: PCC {stats['pcc']:.7f} < {PCC_MIN}")
            if stats["max_rel"] > MAX_REL:
                failures.append(f"{name}: max_rel {stats['max_rel']:.3e} > {MAX_REL}")
            log(
                f"{name:<48} 1 - pcc {1 - stats['pcc']:.1e}  max_abs {stats['max_abs']:.3e}  "
                f"max_rel {stats['max_rel']:.3e}  {'ok' if ok else 'FAIL'}"
            )

        beat.stage = "loading the reference"
        log(beat.stage)
        model = ReferenceModel.from_pretrained(cfg)

        beat.stage = "reference vision tower"
        log(beat.stage)
        with torch.no_grad():
            encoded = model.vision.encode(inp.patches, inp.image_grid_thw)
            check("vision encoder (post_layernorm)", encoded, off["vision_encoder"])
            check("vision tower + projector", model.vision.project(encoded), off["vision"])
        del encoded

        beat.stage = "reference prefill"
        log(beat.stage)
        logits, prompt_cache = model.prefill(inp)
        if tuple(off["cases"]) != CASES:
            failures.append(f"dump has cases {list(off['cases'])}, expected {list(CASES)}")
        long_cot = off["cases"].get("long_forced", {}).get("forced_cot")
        if long_cot is None or not torch.equal(long_cot, driving_text_cot(cfg)):
            failures.append("the dump's long_forced CoT is not driving_text_cot(cfg)")
        for name, case in off["cases"].items():
            check(f"[{name}] prefill last logits", logits, case["step_logits"][:1])
            check_case(name, case, model, inp, prompt_cache, check, failures, beat)

    report = {"versions": off["versions"], "inputs": exact, "tensors": rows}
    report["cases"] = {name: case["cot_source"] for name, case in off["cases"].items()}
    (dump_dir / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {dump_dir / 'compare.json'}")
    if len(rows) != EXPECTED_CHECKS:
        failures.append(f"ran {len(rows)} tensor checks, expected {EXPECTED_CHECKS}")
    if failures:
        log("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    log(f"PASSED: inputs bitwise equal, {len(rows)} tensors with PCC >= {PCC_MIN} and max_rel <= {MAX_REL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dump", help="run the official stack (official venv)").add_argument(
        "--out", type=Path, required=True
    )
    sub.add_parser("compare", help="check the reference against a dump (repo venv)").add_argument(
        "--dump", type=Path, required=True
    )
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.command == "dump":
        dump(args.out)
    else:
        compare(args.dump)


if __name__ == "__main__":
    main()
