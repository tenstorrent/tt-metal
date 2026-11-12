# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose decoder stages against actual captured CUDA stage inputs/outputs.

Each stage is checked both independently and in a cumulative TT decoder chain.
Affine fits are diagnostics only: outputs are never rescaled for acceptance.
"""

import argparse
import json
from pathlib import Path


from ..config import load_config
from .manifest import load_manifest, materialize_snapshot
from .metrics import affine_diagnostics, compare_tensors
from .replay_component import checkpoint_state, runtime_version


def diagnose(args):
    import ttnn
    from ..tt.audio_vae import AudioVAEConfig, TtAudioVAE
    from ..tt.ops import upload

    manifest = load_manifest(args.reference)
    config = load_config(args.checkpoint)
    state = checkpoint_state(args.checkpoint, manifest, codec=True)
    dtype = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}[args.dtype]
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=262144)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dtype": args.dtype,
        "conv_operand_dtype": "bfloat16",
        "ttnn_version": runtime_version(),
        "ttnn_module": str(ttnn.__file__),
        "stages": records,
    }

    def persist():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def to_device(value):
        return upload(value.transpose(1, 2).unsqueeze(1).contiguous(), device, dtype)

    def to_host(value):
        return ttnn.to_torch(value).squeeze(1).transpose(1, 2).contiguous()

    def compare(expected, actual):
        return {**compare_tensors(expected, actual).to_dict(), **affine_diagnostics(expected, actual)}

    try:
        model = TtAudioVAE(state, "", device, dtype, AudioVAEConfig.from_mapping(config.get("audio_vae_config") or {}))
        events = {}
        for event in manifest["events"]:
            if event["component"].startswith("audio_vae.decoder.model."):
                events.setdefault(event["component"], event)
        chain = None
        for index in range(len(model.decoder) + 1):
            name = f"audio_vae.decoder.model.{index}.forward"
            event = events[name]
            inputs = materialize_snapshot(args.reference, manifest, event["args"])
            expected = materialize_snapshot(args.reference, manifest, event["output"])
            isolated = to_device(inputs[0])
            if chain is None:
                chain = to_device(inputs[0])
            elif index in model.conditions:
                chain = model.conditions[index](chain, model.out_sample_rate)
            layer = model.decoder[index] if index < len(model.decoder) else ttnn.tanh

            def apply(value):
                for operation in layer if isinstance(layer, tuple) else (layer,):
                    value = operation(value)
                return value

            isolated = apply(isolated)
            actual_isolated = to_host(isolated)
            ttnn.deallocate(isolated)
            chain = apply(chain)
            actual_chain = to_host(chain)
            record = {
                "stage": name,
                "shape": list(expected.shape),
                "isolated": compare(expected, actual_isolated),
                "cumulative": compare(expected, actual_chain),
            }
            records.append(record)
            persist()
            print(json.dumps(record), flush=True)
        report["complete"] = True
        persist()
    finally:
        ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
