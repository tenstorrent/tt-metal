"""Minimal repro: does init_server_context fail when run inside a multiprocessing.Process child?"""
import multiprocessing as mp
import os
import sys


def child_main():
    # Match the runner's environment exactly.
    os.environ.setdefault("TT_QWEN3_CP_FP32", "1")
    os.environ.pop("TT_MM_THROTTLE_PERF", None)
    os.environ.pop("TT_METAL_CACHE", None)
    os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    # Imports happen INSIDE the child after fork/spawn — same as the worker.
    import ttnn
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import TTSConfig, init_server_context, load_weights
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS

    print("[child] loading weights...", flush=True)
    main_weights, decoder_weights = load_weights()
    print(f"[child] {len(main_weights)} main + {len(decoder_weights)} decoder", flush=True)

    print("[child] opening device...", flush=True)
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=100_000_000, num_command_queues=2)
    device.enable_program_cache()

    print("[child] building model (TT_QWEN3_CP_FP32=" + os.environ.get("TT_QWEN3_CP_FP32", "0") + ")", flush=True)
    model = Qwen3TTS(device=device, state_dict=main_weights)

    config = TTSConfig(max_new_tokens=1500)
    config.repetition_penalty = 1.15

    print("[child] calling init_server_context...", flush=True)
    ctx = init_server_context(device, model, config, main_weights)
    print("[child] init_server_context COMPLETED", flush=True)

    ttnn.close_device(device)
    print("[child] device closed", flush=True)


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "spawn"
    print(f"[main] using start method = {method}", flush=True)
    mp.set_start_method(method, force=True)
    p = mp.Process(target=child_main)
    p.start()
    p.join()
    print(f"[main] child exited with code {p.exitcode}", flush=True)
    sys.exit(p.exitcode)
