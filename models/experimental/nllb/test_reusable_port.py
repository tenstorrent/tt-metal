# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Configurable independent component and standalone text smoke, no fixed CHIA paths."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
import ttnn

if __package__:
    from . import backend
    from .translate import load_text_inputs
    from .nllb_validation import validate_config
else:
    import backend
    from translate import load_text_inputs
    from nllb_validation import validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--tokenizer-directory")
    parser.add_argument("--precision", choices=("bf16", "bfp8_b"), default="bf16")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--source-language", default="eng_Latn")
    parser.add_argument("--target-language", default="fra_Latn")
    parser.add_argument("--text", default="Hello world.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--cli-timeout", type=float, default=180)
    parser.add_argument("--public-inputs", help="Optional public smoke NPZ for tokenizer parity")
    args = parser.parse_args()
    root = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else (root if root.is_dir() else root.parent) / "config.json"
    config = validate_config(json.loads(config_path.read_text()))
    tokenizer, ids, mask, target = load_text_inputs(
        root,
        config,
        args.source_language,
        args.target_language,
        [args.text],
        tokenizer_directory=args.tokenizer_directory,
    )
    if args.public_inputs:
        with np.load(args.public_inputs, allow_pickle=False) as public:
            for field, value in (("input_ids", ids), ("attention_mask", mask)):
                keys = [key for key in public.files if key == field or key.endswith("__" + field)]
                assert len(keys) == 1, (field, public.files)
                assert np.array_equal(value, public[keys[0]]), field
        print("PASS public tokenizer parity", flush=True)
    # Run the standalone child before this process opens TTNN: close_device alone
    # does not destroy the process-global UMD cluster or release its chip lock.
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "translation.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("translate.py")),
            "--checkpoint",
            str(root),
            "--config",
            str(config_path),
            "--device",
            str(args.device),
            "--source-language",
            args.source_language,
            "--target-language",
            args.target_language,
            "--text",
            args.text,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--precision",
            args.precision,
            "--output",
            str(output),
        ]
        if args.tokenizer_directory:
            command.extend(["--tokenizer-directory", args.tokenizer_directory])
        started = time.perf_counter()
        print("PHASE standalone_child_start", flush=True)
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=args.cli_timeout, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            print("CHILD_FAILURE", repr(error), flush=True)
            print("CHILD_STDOUT", error.stdout, flush=True)
            print("CHILD_STDERR", error.stderr, flush=True)
            raise
        print("CLI_PROCESS_SECONDS", time.perf_counter() - started, flush=True)
        print("CHILD_STDERR", child.stderr, flush=True)
        result = json.loads(child.stdout)
        assert result == json.loads(output.read_text()), "stdout/file JSON mismatch"
        print("PASS whole stdout JSON", flush=True)
        tokens = result["token_ids"]
        assert len(tokens) == 1 and tokens[0][:2] == [2, target]
        assert 2 <= len(tokens[0]) <= args.max_new_tokens + 1
        assert result["precision_policy"]["mode"] == args.precision
        assert len(result["translations"]) == 1
        print("STANDALONE_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    generator = torch.Generator().manual_seed(20260921)
    x = torch.randn((1, 1, 32, 32), generator=generator) * 0.1
    w = torch.randn((32, 32), generator=generator) * 0.1
    bias = torch.randn((1, 1, 1, 32), generator=generator) * 0.01
    reference = x @ w.T + bias
    print("PHASE parent_open_after_child_exit", flush=True)
    device = ttnn.open_device(device_id=args.device)
    try:
        component = backend.Backend.__new__(backend.Backend)
        component.device = device
        component.kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        component.weights = {
            "probe.weight": component.upload(w, dtype=ttnn.bfloat8_b if args.precision == "bfp8_b" else ttnn.bfloat16),
            "probe.bias": component.upload(bias),
        }
        actual = ttnn.to_torch(component.linear(component.upload(x), "probe")).float()
        ttnn.synchronize_device(device)
        nrmse = torch.sqrt(torch.mean((actual - reference) ** 2, dim=-1)) / torch.sqrt(
            torch.mean(reference**2, dim=-1)
        ).clamp_min(1e-8)
        assert torch.isfinite(actual).all() and nrmse.max().item() <= 0.04
        print("PASS independent FP32 component", nrmse.max().item(), flush=True)
        del component, actual
        model = backend.create_backend(str(root), config, device, precision=args.precision)
        ttnn.synchronize_device(device)
        started = time.perf_counter()
        actual_tokens = model.generate(ids, mask, target, args.max_new_tokens)
        ttnn.synchronize_device(device)
        print("TOKEN_API_SECONDS", time.perf_counter() - started, flush=True)
        assert actual_tokens.tolist() == tokens, (actual_tokens.tolist(), tokens)
        print("PASS standalone/token API exact parity", flush=True)
        del model
    finally:
        ttnn.close_device(device)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__).with_name(name)
            for name in (
                "backend.py",
                "nllb_validation.py",
                "translate.py",
                "test_checkpoint_loading.py",
                "test_reusable_port.py",
            )
        )
    }
    print("ARTIFACT_HASHES " + json.dumps(hashes), flush=True)


def test_standalone_cli_and_token_api(tmp_path, nllb_device_id):
    """Run in a fresh process so component fixtures cannot retain the CLI chip lock."""
    import os
    import pytest

    precision = os.environ.get("NLLB_TEST_PRECISION", "bf16")
    if precision not in ("bf16", "bfp8_b"):
        raise ValueError("NLLB_TEST_PRECISION must be bf16 or bfp8_b")
    checkpoint = os.environ.get("NLLB_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set NLLB_TEST_CHECKPOINT for actual standalone model inference")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint",
        checkpoint,
        "--device",
        str(nllb_device_id),
        "--precision",
        precision,
    ]
    for variable, option in (("NLLB_TEST_CONFIG", "--config"), ("NLLB_TEST_TOKENIZER", "--tokenizer-directory")):
        if os.environ.get(variable):
            command.extend([option, os.environ[variable]])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


if __name__ == "__main__":
    main()
