# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Observe an oracle-assisted prefix-state control; never acceptance evidence.

Reuses the V5 observer and the verified V6 prefix-control setup unchanged.
"""

import argparse
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v5_layer_stream_probe as observer
from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_prefix_state_control import DOC, cpu_setup, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--positions", type=int, nargs="+", default=[3271])
    parser.add_argument("--include-worst", action="store_true")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-length", type=int, default=8192)
    parser.add_argument("--continuation-length", type=int, default=33)
    parser.add_argument("--oracle-cache", type=Path, default=DOC.parent / "oracle_cache/v5_layer34_seed154_8192.pt")
    parser.add_argument("--prefix-capture", type=Path, default=DOC / "v5_combined_layer34_stream_probe.json.pos128.pt")
    parser.add_argument("--failure-json", type=Path, default=DOC / "v6_inline_ordered_router_prefix_control.json")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists() and not Path(str(args.output) + ".provenance.json").exists()
    canonical, identity = cpu_setup(args)
    identity.update(
        classification="ORACLE_ASSISTED_PREFIX_STATE_TRANSPLANT_STREAM_OBSERVER",
        stream_wrapper_sha256=digest(__file__),
        observer_source_sha256=digest(observer.__file__),
        candidate_selection="Oracle-assisted diagnostic snapshot; never final acceptance",
        acceptance_eligible=False,
    )
    failure = json.loads(args.failure_json.read_text())
    assert failure["acceptance_eligible"] is False
    assert failure["classification"] == "ORACLE_ASSISTED_PREFIX_STATE_TRANSPLANT_DIAGNOSTIC"
    assert failure["measured_decoder_source_sha256"] == identity["measured_decoder_source_sha256"]
    assert failure["canonical_prefix_state_sha256"] == identity["canonical_prefix_state_sha256"]
    original_identity, original_loader = observer.decoder_identity, observer.load_decoder
    original_stream, original_save = observer.load_stream, torch.save
    injections = []

    def decoder_identity(options):
        return {**original_identity(options), **identity}

    def load_stream(options):
        cached, report = original_stream(options)
        report.update(identity, oracle_assisted=True, prefix_state_injections=injections)
        report["source_sha256"].update(
            {
                str(Path(__file__).resolve()): digest(__file__),
                str(Path(cpu_setup.__code__.co_filename).resolve()): identity["wrapper_sha256"],
            }
        )
        return cached, report

    def load_decoder(options, report):
        snapshot_class = original_loader(options, report)
        import ttnn

        class PrefixStreamControl(snapshot_class):
            @classmethod
            def from_state_dict(cls, *a, **kw):
                instance = super().from_state_dict(*a, **kw)
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

        return PrefixStreamControl

    def save(value, path, *a, **kw):
        if isinstance(value, dict) and "hf_values" in value and "tt_values" in value:
            value = {**value, **identity, "oracle_assisted": True, "prefix_state_injections": list(injections)}
        return original_save(value, path, *a, **kw)

    observer.decoder_identity, observer.load_decoder = decoder_identity, load_decoder
    observer.load_stream, torch.save = load_stream, save
    try:
        observer.run(args)
        assert args.cpu_only or len(injections) == 1
    finally:
        observer.decoder_identity, observer.load_decoder = original_identity, original_loader
        observer.load_stream, torch.save = original_stream, original_save
        assert digest(args.snapshot) == identity["measured_decoder_source_sha256"]
        assert digest(observer.__file__) == identity["observer_source_sha256"]


if __name__ == "__main__":
    main()
