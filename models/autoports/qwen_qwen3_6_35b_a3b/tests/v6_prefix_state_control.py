# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Oracle-assisted prefix-state transplant diagnostic; never acceptance evidence.

Keeps the shared stress harness unchanged. Uploads canonical prefix state during
setup, copies it after the first prefill, then uses the candidate's actual decode.
"""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import torch

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_setup(args):
    from transformers import DynamicCache

    from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

    torch.set_num_threads(8)
    oracle = torch.load(args.oracle_cache, map_location="cpu", weights_only=False)
    capture = torch.load(args.prefix_capture, map_location="cpu", weights_only=False)
    key = oracle["key"]
    expected = {k: getattr(args, k) for k in ("layer", "seed", "prompt_length", "decode_length", "continuation_length")}
    assert key["parameters"] == expected
    assert key["revision"] == reference.REVISION and key["torch_version"] == torch.__version__
    for name, sha in key["source_hashes"].items():
        assert digest(name) == sha, name
    assert capture["position"] == args.prompt_length
    assert capture["layer"] == args.layer and capture["seed"] == args.seed
    assert capture["stream_sha256"]["prefix"] == tensor_digest(oracle["prefix"])
    assert capture["stream_sha256"]["tokens"] == tensor_digest(oracle["tokens"])
    hf = reference.layer(reference.weights(args.layer), args.layer)
    assert hf.layer_type == "linear_attention"
    cache = DynamicCache(config=reference.config())
    prefix_result = reference.hf_forward(hf, oracle["prefix"], cache, 0)
    assert torch.equal(prefix_result, oracle["ref_p"]), "Fresh HF prefix differs from immutable oracle"
    canonical = cache.layers[args.layer].recurrent_states.clone()
    assert torch.equal(canonical, capture["hf_values"]["delta_state_input"])
    identity = dict(
        classification="ORACLE_ASSISTED_PREFIX_STATE_TRANSPLANT_DIAGNOSTIC",
        acceptance_eligible=False,
        measured_decoder_source=str(args.snapshot.resolve()),
        measured_decoder_source_sha256=digest(args.snapshot),
        wrapper_sha256=digest(__file__),
        oracle_cache_sha256=digest(args.oracle_cache),
        prefix_capture_sha256=digest(args.prefix_capture),
        prefix_sha256=tensor_digest(oracle["prefix"]),
        canonical_prefix_state_sha256=tensor_digest(canonical),
        fresh_hf_prefix_and_state_bit_exact=True,
        injection="One setup-uploaded canonical HF recurrent state copied after first candidate prefill; conv state unchanged",
    )
    return canonical, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-length", type=int, default=8192)
    parser.add_argument("--continuation-length", type=int, default=33)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--oracle-cache", type=Path, default=DOC.parent / "oracle_cache/v5_layer34_seed154_8192.pt")
    parser.add_argument("--prefix-capture", type=Path, default=DOC / "v5_combined_layer34_stream_probe.json.pos128.pt")
    parser.add_argument("--cpu-setup-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve existing evidence"
    canonical, identity = cpu_setup(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_setup_only:
        args.output.write_text(json.dumps(dict(identity, status="CPU_SETUP_PASS", hardware_run=False), indent=2) + "\n")
        print("CPU_SETUP_PASS", args.output)
        return

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import stress_fused_decoder as stress

    snapshot_class = runpy.run_path(str(args.snapshot.resolve()))["FusedDecoder"]
    original_record, original_decoder = stress.record, stress.FusedDecoder
    injections = []

    class PrefixStateControl(snapshot_class):
        @classmethod
        def from_state_dict(cls, *a, **kw):
            instance = super().from_state_dict(*a, **kw)
            # Setup outside shared harness device_only regions and trace capture.
            instance.control_state = ttnn.from_torch(
                canonical.float(),
                device=kw["mesh_device"],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            instance.control_injected = False
            return instance

        def prefill_forward(self, hidden_states, *, state, plan):
            result = super().prefill_forward(hidden_states, state=state, plan=plan)
            if not self.control_injected:
                assert hidden_states.shape[1] == args.prompt_length and plan[0][2] == 0
                ttnn.copy(self.control_state, state["recurrent"])
                self.control_injected = True
                injections.append(dict(after_prefill_length=args.prompt_length, state="recurrent"))
            return result

    def record(path):
        original_record(path)
        provenance = Path(str(path) + ".provenance.json")
        data = json.loads(provenance.read_text())
        data.update(identity)
        provenance.write_text(json.dumps(data, indent=2) + "\n")

    stress.FusedDecoder, stress.record = PrefixStateControl, record
    try:
        stress.run(args)
    finally:
        stress.FusedDecoder, stress.record = original_decoder, original_record
        assert digest(args.snapshot) == identity["measured_decoder_source_sha256"]
        if args.output.exists():
            data = json.loads(args.output.read_text())
            data.update(identity, injections=injections, hardware_run=True)
            args.output.write_text(json.dumps(data, indent=2) + "\n")
        assert len(injections) <= 1


if __name__ == "__main__":
    main()
