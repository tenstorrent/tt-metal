import os
import time
import pytest

try:
    from diffusers import DiffusionPipeline
except Exception:
    DiffusionPipeline = None


@pytest.mark.parametrize(
    "model_id",
    [
        # Default to a Wan2.2 T2V diffusers checkpoint if available; override via WAN_T2V_MODEL_ID env.
        os.environ.get("WAN_T2V_MODEL_ID", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
    ],
)
def test_text_encoder_inference_cpu(model_id: str) -> None:
    import os, torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "32")))
    torch.set_num_interop_threads(int(os.environ.get("TORCH_INTEROP_THREADS", "32")))

    print("OMP_NUM_THREADS =", os.getenv("OMP_NUM_THREADS"))
    print("MKL_NUM_THREADS =", os.getenv("MKL_NUM_THREADS"))
    print("torch.get_num_threads() =", torch.get_num_threads())
    print("torch.get_num_interop_threads() =", torch.get_num_interop_threads())
    print("os.cpu_count() =", os.cpu_count())

    # Optional: check affinity if taskset is available
    import subprocess, os as _os

    try:
        print(
            "taskset:",
            subprocess.check_output(["taskset", "-cp", str(_os.getpid())]).decode(),
        )
    except Exception as e:
        print("taskset check failed:", e)

    if DiffusionPipeline is None:
        pytest.skip("diffusers is not installed")

    # Load pipeline on CPU
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.to("cpu")

    prompt = "A cute corgi playing guitar on stage, cinematic lighting, 4k detail"

    # Resolve tokenizer and text encoder
    tokenizer = getattr(pipe, "tokenizer", None)
    text_encoder = getattr(pipe, "text_encoder", None)

    # Some pipelines may expose dual encoders; prefer the primary if present
    if tokenizer is None or text_encoder is None:
        tokenizer = getattr(pipe, "tokenizer_1", tokenizer)
        text_encoder = getattr(pipe, "text_encoder_1", text_encoder)
    if tokenizer is None or text_encoder is None:
        tokenizer = getattr(pipe, "tokenizer_2", tokenizer)
        text_encoder = getattr(pipe, "text_encoder_2", text_encoder)

    if tokenizer is None or text_encoder is None:
        raise RuntimeError("Could not locate tokenizer/text_encoder on the loaded pipeline.")

    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.get("input_ids", None)
    attention_mask = inputs.get("attention_mask", None)
    if input_ids is None:
        raise RuntimeError("Tokenizer did not return input_ids.")

    input_ids = input_ids.to("cpu")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")

    # Time only the text encoder forward pass
    start_time = time.time()
    with torch.no_grad():
        # Support common encoder call signatures (e.g., CLIP/T5)
        _ = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    elapsed_s = time.time() - start_time

    print(f"Text encoder inference time (CPU): {elapsed_s:.4f} seconds")
