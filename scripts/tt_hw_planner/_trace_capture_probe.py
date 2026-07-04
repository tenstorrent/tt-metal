from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_LADDER = [
    (
        "residency",
        r"per-layer weight|load\+evict|load/evict|lazy per-layer|does not fit|stream[^\n]{0,24}(weight|layer|expert)"
        r"|evict[^\n]{0,24}(weight|layer|expert)|(weight|layer|expert)[^\n]{0,24}evict",
        "Weights stream from host per layer (build/evict). Make them RESIDENT: shard the parameter-heavy "
        "layers across the mesh so each chip holds its slice for the whole run (for an MoE that is "
        "expert-parallel via get_shard_plan under the host-free goal; for a dense model, higher TP or "
        "pipeline-parallel), and build every layer ONCE at init instead of build-then-evict. The runtime "
        "op-signature probe is the authoritative generic check: many distinct host->device weight uploads "
        "during a forward == streaming.",
    ),
    (
        "token_feed",
        r"torch\.cat\s*\(|from_torch\([^)]*ids|\.item\(\)",
        "Decode feeds the next token via host (torch.cat / re-upload / .item()). Feed it ON DEVICE: write "
        "the ttnn.argmax result into a fixed [1,1] input tensor on device and advance an on-device position "
        "index; do not rebuild the input on host.",
    ),
    (
        "kv_cache",
        None,
        "No persistent KV cache -> full re-prefill per token. Add an on-device KV cache (persistent K/V, "
        "single-token append at the current position); see models/tt_transformers/tt/attention.py.",
    ),
    (
        "decode_step",
        None,
        "Expose the GENERIC decode contract the perf/2CQ engine binds to (perf_adapter), and trace it: "
        "(1) decode_prefill(input_ids)->state seeds the resident KV/SSM once (and any cross-attention KV "
        "for a seq2seq decoder); (2) decode_step(state)->state runs exactly ONE fixed-shape, host-op-free "
        "token (on-device argmax feed, constant [1,1] shapes every step) and returns the advanced state; "
        "(3) decode_write_inputs(state)->None stages the NEXT token on command-queue 1 -- this hook is what "
        "flips the engine into the trace+2CQ path. Then trace_capture_selftest(device) must wrap a "
        "decode_step in ttnn.begin_trace_capture / end_trace_capture and replay it. Use these EXACT public "
        "names: they are the model-agnostic seam the optimize/2CQ tool reads; the BODY is yours to write "
        "for this architecture.",
    ),
]


def _read(demo_dir: Path) -> str:
    text = []
    for sub in ("tt", "_stubs"):
        d = demo_dir / sub
        if d.is_dir():
            for p in sorted(d.glob("*.py")):
                try:
                    text.append(p.read_text())
                except OSError:
                    pass
    return "\n".join(text)


_AR_MARKERS = (
    r"def generate\b|max_new_tokens|\bnext_token\b|for\s+\w+\s+in\s+range\([^)]*max_new|_sample_tt|"
    r"argmax\([^)]*dim|decode_prefill|autoregress"
)


def _stage_names(src: str) -> list:
    m = re.search(r"PIPELINE_STAGES\s*=\s*\[([^\]]*)\]", src)
    if not m:
        return []
    return re.findall(r"[\"']([A-Za-z0-9_]+)[\"']", m.group(1))


def _stage_contract_ok(src: str, stage: str) -> bool:
    if stage == "decode":
        return bool(re.search(r"def\s+decode_step\s*\(", src)) and bool(
            re.search(r"def\s+decode_write_inputs\s*\(", src)
        )
    return all(
        bool(re.search(r"def\s+%s_%s\s*\(" % (re.escape(stage), suf), src))
        for suf in ("trace_setup", "trace_step", "write_inputs")
    )


def probe(demo_dir: Path) -> dict:
    src = _read(demo_dir)
    lad = {name: (pat, guidance) for name, pat, guidance in _LADDER}
    is_ar = bool(re.search(_AR_MARKERS, src))
    blockers = []
    stages = _stage_names(src)

    if re.search(lad["residency"][0], src):
        blockers.append({"rung": "residency", "guidance": lad["residency"][1]})

    if is_ar:
        if re.search(lad["token_feed"][0], src):
            blockers.append({"rung": "token_feed", "guidance": lad["token_feed"][1]})
        has_kv = re.search(r"kv[_ ]?cache|update_cache|paged_update|cache_k\b|past_key|kvcache", src, re.I)
        reprefill = re.search(r"for .* in range\(.*max_new|re-?prefill|full recompute", src, re.I)
        if reprefill and not has_kv:
            blockers.append({"rung": "kv_cache", "guidance": lad["kv_cache"][1]})

    has_trace_hook = ("begin_trace_capture" in src) and bool(re.search(r"def\s+trace_capture_selftest", src))
    if is_ar:
        has_contract = bool(re.search(r"def\s+decode_step\s*\(", src)) and bool(
            re.search(r"def\s+decode_write_inputs\s*\(", src)
        )
        if not (has_trace_hook and has_contract):
            blockers.append({"rung": "trace_entry", "guidance": lad["decode_step"][1]})
    elif not has_trace_hook:
        g = (
            "No trace-capturable entry for this feed-forward model. The trace UNIT is the steady-state "
            "FORWARD pass (not a decode step): make its input shapes FIXED and its forward host-op-free, "
            "then expose trace_capture_selftest(device) that wraps ONE forward in "
            "ttnn.begin_trace_capture / end_trace_capture, and (forward-2CQ) a forward_step(inputs) + "
            "write_inputs(inputs) CQ1 staging hook so the generic engine can overlap the next input "
            "upload with compute. For a multi-stage pipeline (encoder / decoder / vocoder, e.g. Seamless) "
            "do this PER STAGE — an AR decoder stage additionally needs the decode-step / KV rungs, a "
            "conv / vocoder stage only needs residency + fixed-shape forward."
        )
        blockers.append({"rung": "trace_entry", "guidance": g})

    missing_stage = [s for s in stages if not _stage_contract_ok(src, s)]
    if stages and missing_stage:
        g = (
            "PIPELINE_STAGES declares %s but these stages lack the trace+2CQ contract: %s. "
            "Emit per stage (COMMAND 3): a one-shot stage needs <stage>_trace_setup / "
            "<stage>_trace_step / <stage>_write_inputs (pin the variable dim to a fixed C, hoist "
            "the shape-dependent constants from the HF reference OUTSIDE the trace); an AR decode "
            "stage needs decode_step + decode_write_inputs (resident self-/cross-attn KV)." % (stages, missing_stage)
        )
        blockers.append({"rung": "trace_stage", "guidance": g})

    trace_ready = not blockers
    device_capture = None
    if trace_ready:
        device_capture = _device_selftest(demo_dir)

    return {
        "model_class": "autoregressive" if is_ar else "feed_forward",
        "trace_ready": bool(trace_ready and (device_capture is None or device_capture.get("ok"))),
        "static_blockers": blockers,
        "device_capture": device_capture,
        "stages": stages,
    }


def _device_selftest(demo_dir: Path) -> dict | None:
    try:
        sys.path.insert(0, str(demo_dir.parent))
        mod = __import__("%s.tt.pipeline" % demo_dir.name, fromlist=["pipeline"])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "cannot import tt.pipeline: %s" % e}
    fn = getattr(mod, "trace_capture_selftest", None)
    if fn is None:
        return {"ok": False, "reason": "no trace_capture_selftest hook to run a real device capture"}
    try:
        ok = bool(fn())
        return {"ok": ok, "reason": "" if ok else "trace_capture_selftest returned False"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "begin_trace_capture raised (host op in decode step): %s" % e}


def main(argv):
    demo_dir = Path(argv[1]).resolve()
    r = probe(demo_dir)
    print("TRACE_PROBE=" + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
