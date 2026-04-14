# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the TT-only RVC performant runner.

Example:
    export RVC_CONFIGS_DIR="$PWD/models/demos/rvc/data/configs"
    export RVC_ASSETS_DIR="$PWD/models/demos/rvc/data/assets"

    ./python_env/bin/python models/demos/rvc/scripts/run_performant_runner.py \
      --num-secs 3.0 \
      --device-id 0 \
      --warmup-runs 1 \
      --iters 5
"""

import argparse
import time

import ttnn
from models.demos.rvc.runner.performant_runner import RVCRunner
from models.demos.rvc.runner.performant_runner_infra import RVCInferenceConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TT-only RVC performant runner.")
    parser.add_argument("--num-secs", type=float, default=3.0, help="Prepared input audio duration at 16 kHz.")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID.")
    parser.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument("--f0-method", default="pm", choices=["pm"], help="F0 method.")
    parser.add_argument("--index-rate", type=float, default=0.75, help="Index rate.")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25, help="RMS mix rate.")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect rate.")
    parser.add_argument("--device-id", type=int, default=0, help="TT device id.")
    parser.add_argument("--l1-small-size", type=int, default=65384, help="CreateDevice l1_small_size.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--iters", type=int, default=5, help="Timed inference iterations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = RVCRunner()
    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=args.l1_small_size)
    try:
        inference_config = RVCInferenceConfig(
            num_secs=args.num_secs,
            speaker_id=args.speaker_id,
            f0_up_key=args.f0_up_key,
            f0_method=args.f0_method,
            index_rate=args.index_rate,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
        )
        runner.initialize_inference(device, {"inference": inference_config, "warmup_runs": args.warmup_runs})
        torch_input_tensor, _ = runner.test_infra.setup_l1_sharded_input(device)

        start_time = time.time()
        output = None
        for _ in range(args.iters):
            output = runner.run(torch_input_tensor)
        ttnn.synchronize_device(device)
        end_time = time.time()

        output_np = runner.test_infra._to_numpy(output)
        passed, message = runner.validate()
        avg_sec = (end_time - start_time) / args.iters
        print(f"avg_sec={avg_sec:.6f}")
        print(f"num_samples={len(output_np)}")
        print(f"num_input_samples={torch_input_tensor.shape[0]}")
        print(f"passed={passed}")
        print(f"message={message}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
