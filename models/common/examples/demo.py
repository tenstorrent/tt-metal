"""
TTTv2 demo scaffolding.

This file sketches the standard pattern for bringing up a model on TT devices
using TTNN and (optionally) TTTv2 modules:

1. Open a mesh device based on the target system.
2. Configure the global validation registry.
3. Build a model that runs on the mesh device (ideally composed from TTTv2
   building blocks plus any model-specific adapters).
4. Prepare a prompt or input batch.
5. Run a short generation / inference pass.
6. Print a validation report (accuracy + performance) if any validations ran.
7. Close the device.

Model authors are expected to copy or import this pattern and fill in the
model-specific pieces (weight loading, tokenizer, adapters, etc.).
"""

from __future__ import annotations

import os
from typing import Any

import ttnn
from models.common.validation_tools import get_validation_registry


def open_mesh_device_from_env() -> ttnn.MeshDevice:
    """
    Open a TTNN mesh device based on the MESH_DEVICE environment variable.

    This mirrors the pattern used in ds_r1_qwen.py and other demos:
    - MESH_DEVICE=N150  -> MeshShape([1, 1])
    - MESH_DEVICE=N300  -> MeshShape([1, 2])
    - MESH_DEVICE=T3K   -> MeshShape([1, 8])
    - MESH_DEVICE=T3K2x4 -> MeshShape([2, 4])
    - default           -> use first available device as a 1x1 mesh
    """
    mesh_env = os.environ.get("MESH_DEVICE")
    if mesh_env == "N150":
        mesh_shape = ttnn.MeshShape([1, 1])
    elif mesh_env == "N300":
        mesh_shape = ttnn.MeshShape([1, 2])
    elif mesh_env == "T3K":
        mesh_shape = ttnn.MeshShape([1, 8])
    elif mesh_env == "T3K2x4":
        mesh_shape = ttnn.MeshShape([2, 4])
    else:
        device_ids = ttnn.get_device_ids()
        if not device_ids:
            raise RuntimeError("No TT devices found")
        # Simple default: single-device mesh
        mesh_shape = ttnn.MeshShape([1, 1])

    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    # Set the default device so compare_to_ttnn / auto-compose helpers work
    ttnn.SetDefaultDevice(mesh_device)

    print(f"Using {mesh_device.get_num_devices()} device(s)")
    print()
    return mesh_device


def configure_validation() -> None:
    """
    Enable or disable validation for this demo run.

    By default we enable validation, but this can be controlled via an env var
    if desired (e.g., TTTV2_VALIDATE=0 to disable).
    """
    registry = get_validation_registry()
    env_value = os.environ.get("TTTV2_VALIDATE")
    if env_value is None:
        enabled = True
    else:
        enabled = env_value not in ("0", "false", "False")

    registry.enabled = enabled
    if enabled:
        print("Validation registry enabled – decorated functions will record results.")
    else:
        print("Validation registry disabled – decorated functions will not record results.")
    print()


def build_model(mesh_device: ttnn.MeshDevice) -> Any:
    """
    Construct and return the model used in this demo.

    Typical TTTv2 pattern (not implemented here):
    - Map reference weights (e.g., from a HF model) into TTNN/TTTv2 layouts.
    - Instantiate TTTv2 modules (attention, MLP, normalization, etc.) with
      hardware-aware configuration.
    - Compose them into higher-level blocks such as DecoderLayer.
    - Optionally wrap them in thin HF-style adapters that preserve the
      original forward(...) signature.

    This function is intentionally left as a placeholder; model authors should
    replace its body with their model-specific construction code.
    """
    # TODO: replace with a real model implementation built from TTTv2 modules.
    print("build_model(mesh_device): please replace this placeholder with your model.")
    return None


def prepare_prompt() -> str:
    """
    Prepare a prompt or input batch for the demo.

    In real demos this might:
    - Use a HF tokenizer's chat template.
    - Build a batch of sequences for throughput testing.

    Here we just use a simple string placeholder.
    """
    return "What is 2+2? Answer briefly."


def generate(model: Any, prompt: str, max_new_tokens: int = 32) -> str:
    """
    Run a short generation or inference pass.

    The exact signature will depend on the model; the ds_r1_qwen.py example
    uses a `generate(model, tokenizer, prompt, max_new_tokens=...)` helper.
    For this generic TTTv2 demo we simply return a placeholder string to keep
    the file runnable even before a concrete model is wired in.
    """
    if model is None:
        return "<demo placeholder: no model is configured yet>"

    # Example pattern (not executed here):
    #
    # tokens = tokenizer(prompt, return_tensors="pt")
    # response = model.generate(tokens, max_new_tokens=max_new_tokens)
    # return tokenizer.decode(response[0], skip_special_tokens=True)
    #
    # Replace the placeholder below with your real generation call.
    return "<demo placeholder response from model>"


def maybe_print_validation_report() -> None:
    """
    Print a summary validation report if any validations ran.

    This uses the global ValidationRegistry. If no decorated functions were
    called, the registry will be empty and we do nothing.
    """
    registry = get_validation_registry()
    if registry.results:
        registry.print_report(verbose=True)


def main(max_new_tokens: int = 32) -> None:
    print("TTTv2 demo – device bring-up + validation pattern")
    print("=" * 80)
    print()

    mesh_device = open_mesh_device_from_env()
    try:
        configure_validation()

        # Build model (replace placeholder with real TTTv2-based model).
        model = build_model(mesh_device)

        # Prepare prompt / inputs.
        prompt = prepare_prompt()
        print("Prompt:", prompt)
        print()

        # Run generation/inference.
        print("Generating...")
        response = generate(model, prompt, max_new_tokens=max_new_tokens)
        print()
        print("Response:", response)
        print()

        # Print validation report if any validations were run.
        maybe_print_validation_report()
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
