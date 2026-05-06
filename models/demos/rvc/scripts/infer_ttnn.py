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
import os
import time

import ttnn
from models.demos.rvc.evals import (
    DEFAULT_TOKEN_ACCURACY_THRESHOLD,
    DEFAULT_WHISPER_MODEL,
    compute_token_accuracy,
    count_whisper_token,
)
from models.demos.rvc.evals.speaker_similarity import compute_speaker_similarity
from models.demos.rvc.runner.performant_runner import RVCInferenceConfig, RVCRunner
from models.demos.rvc.torch_impl.vc.pipeline import Pipeline as TorchPipeline
from models.demos.rvc.utils.f0 import F0Method
from tests.ttnn.utils_for_testing import assert_with_pcc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TT-only RVC performant runner.")
    parser.add_argument("-o", "--output", required=False, help="Output audio path (wav).")
    parser.add_argument("--num-secs", type=float, default=30.0, help="Prepared input audio duration at 16 kHz.")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID.")
    parser.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones.")
    parser.add_argument(
        "--f0-method", default="rapt", choices=["rapt", "dio", "harvest", "crepe", "rmvpe"], help="F0 method."
    )
    parser.add_argument("--index-rate", type=float, default=0.75, help="Index rate.")
    parser.add_argument("--file-index", default=None, help="Optional FAISS feature index path.")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25, help="RMS mix rate.")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect rate.")
    parser.add_argument("--device-id", type=int, default=0, help="TT device id.")
    parser.add_argument("--l1-small-size", type=int, default=65384, help="CreateDevice l1_small_size.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--iters", type=int, default=3, help="Timed inference iterations.")
    parser.add_argument(
        "--count-token", action="store_true", help="Transcribe output audio with Whisper and count transcript tokens."
    )
    parser.add_argument(
        "--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper model identifier for --count-token."
    )
    parser.add_argument("--whisper-device", default="cpu", help="Execution device for Whisper.")
    parser.add_argument(
        "--compute-embedding-similarity",
        action="store_true",
        help="Compute speaker embedding cosine similarity between input and output audio.",
    )
    parser.add_argument("--speaker-similarity-device", default="cpu", help="Execution device for speaker similarity.")
    parser.add_argument(
        "--check-torch-token-accuracy",
        action="store_true",
        help="Compare TTNN output token accuracy against a Torch reference output.",
    )
    parser.add_argument(
        "--segmented-pcc-window-sec",
        type=float,
        default=0.5,
        help="Window size in seconds for --compute-segmented-pcc.",
    )
    parser.add_argument(
        "--segmented-pcc-hop-sec",
        type=float,
        default=0.25,
        help="Hop size in seconds for --compute-segmented-pcc.",
    )
    parser.add_argument(
        "--performance-runner",
        action="store_true",
        help="Use the performance runner.",
    )
    parser.add_argument(
        "--token-accuracy-threshold",
        type=float,
        default=DEFAULT_TOKEN_ACCURACY_THRESHOLD,
        help="Minimum token accuracy required for --check-torch-token-accuracy.",
    )
    return parser.parse_args()


def torch_infer(
    *,
    speaker_id: int,
    f0_up_key: int,
    f0_method: F0Method | str,
    index_rate: float,
    file_index: str | None,
    rms_mix_rate: float,
    protect: float,
    validation=False,
):
    pipe = TorchPipeline(
        if_f0=True,
        version="v1",
        num="48k",
        speaker_id=speaker_id,
        f0_up_key=f0_up_key,
        f0_method=F0Method.from_str(f0_method),
        index_rate=index_rate,
        file_index=file_index,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        validation=validation,
    )
    return pipe.infer()


def main() -> None:
    args = parse_args()
    if not os.getenv("RVC_CONFIGS_DIR"):
        raise RuntimeError("RVC_CONFIGS_DIR is not set.")
    if not os.getenv("RVC_ASSETS_DIR"):
        raise RuntimeError("RVC_ASSETS_DIR is not set.")

    validation = args.check_torch_token_accuracy
    runner = RVCRunner()
    inference_config = RVCInferenceConfig(
        num_secs=args.num_secs,
        speaker_id=args.speaker_id,
        f0_up_key=args.f0_up_key,
        f0_method=F0Method.from_str(args.f0_method),
        index_rate=args.index_rate,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )
    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=args.l1_small_size, num_command_queues=2)
    runner.initialize_inference(
        device, {"inference": inference_config}, validation=validation, performance_runner=args.performance_runner
    )
    torch_input_tensor = runner.ttnn_pipeline.prepare_audio_input()

    ttnn.synchronize_device(device)

    start_time = time.time()
    output = None
    for _ in range(args.iters):
        output = runner.run(torch_input_tensor)
    ttnn.synchronize_device(device)
    end_time = time.time()

    output_np = output.cpu().numpy()
    print(f"output shape: {output_np.shape}, dtype: {output_np.dtype}, device: {output.device}")

    avg_sec = (end_time - start_time) / args.iters
    print(f"Average inference time: {avg_sec:.6f} seconds.")

    if args.output:
        import soundfile as sf

        sf.write(args.output, output_np[0], runner.ttnn_pipeline.tgt_sr, subtype="PCM_16")

    if args.count_token:
        transcript, num_tokens = count_whisper_token(
            output[0],
            runner.ttnn_pipeline.tgt_sr,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
        )
        print(f"whisper_transcript={transcript}")
        print(f"num_whisper_tokens={num_tokens}")
        print(f"tokens/second={num_tokens / avg_sec:.2f}")

    if args.compute_embedding_similarity:
        speaker_similarity = compute_speaker_similarity(
            torch_input_tensor[0],
            output[0],
            reference_sample_rate=runner.ttnn_pipeline.sr,
            candidate_sample_rate=runner.ttnn_pipeline.tgt_sr,
            device=args.speaker_similarity_device,
        )
        print(f"speaker_similarity={speaker_similarity:.6f}")
        print(f"speaker_similarity_percent={speaker_similarity * 100:.2f}")
        print(f"speaker_similarity_pass={str(speaker_similarity > 0.75).lower()}")

    if args.check_torch_token_accuracy:
        torch_output = torch_infer(
            speaker_id=args.speaker_id,
            f0_up_key=args.f0_up_key,
            f0_method=F0Method.from_str(args.f0_method),
            index_rate=args.index_rate,
            file_index=args.file_index,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            validation=True,
        )
        assert tuple(output_np[0].shape) == tuple(
            torch_output.shape
        ), f"Shape mismatch between TTNN output {output.shape} and Torch output {torch_output.shape}"

        msg = assert_with_pcc(torch_output, output.cpu()[0], pcc=args.token_accuracy_threshold)
        print(f"Token accuracy check PCC: {msg}")
        token_accuracy_result = compute_token_accuracy(
            torch_output,
            # torch_output,
            output[0],
            reference_sample_rate=runner.ttnn_pipeline.tgt_sr,
            candidate_sample_rate=runner.ttnn_pipeline.tgt_sr,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            threshold=args.token_accuracy_threshold,
        )
        print(f"reference_transcript={token_accuracy_result.reference_transcript}")
        print(f"candidate_transcript={token_accuracy_result.candidate_transcript}")
        print(f"reference_num_tokens={token_accuracy_result.reference_num_tokens}")
        print(f"candidate_num_tokens={token_accuracy_result.candidate_num_tokens}")
        print(f"token_edit_distance={token_accuracy_result.token_edit_distance}")
        print(f"token_accuracy={token_accuracy_result.token_accuracy:.6f}")
        print(f"token_accuracy_percent={token_accuracy_result.token_accuracy * 100:.2f}")
        print(f"token_accuracy_pass={str(token_accuracy_result.passed).lower()}")

    output_duration_sec = output_np.shape[1] / runner.ttnn_pipeline.tgt_sr
    rtf = avg_sec / output_duration_sec if output_duration_sec > 0 else float("inf")
    print(f"avg_sec={avg_sec:.6f}")
    print(f"output_duration_sec={output_duration_sec:.6f}")
    print(f"rtf={rtf:.6f}")
    print(f"output_shape={output_np.shape}")
    print(f"batch_size={torch_input_tensor.shape[0]}")
    print(f"num_input_samples={torch_input_tensor.shape[1]}")


if __name__ == "__main__":
    main()
