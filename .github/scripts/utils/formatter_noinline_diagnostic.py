# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One diagnostic of the original merge-native Llama unit recipe; no CI waiver."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import yaml

BASE = "2a631e7214d4e3aa6440ec50048edd278e88422d"
BINDING = "5e96117a53ceb59cd2e6d067004ac30671d801f2281ca4a624acf61c6473c729"
WHEEL = "e1bbf10ba5123d767e940e6e830cdc3c6e2ad0aef5bbb6bc158ead590cdfd09b"
OUT = Path("generated/noinline-diagnostic")
RECIPE = """export TT_METAL_WATCHER=15
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct
pytest --timeout 120 models/tt_transformers/tests/test_warm_cache_marker.py
pytest --timeout 60 models/tt_transformers/tests/test_decode_output_rows.py
pytest --timeout 300 models/tt_transformers/tests/test_traced_prefill_rope_slice.py
pytest --timeout 300 models/tt_transformers/tests/test_embedding.py
pytest --timeout 300 models/tt_transformers/tests/test_rms_norm.py
pytest --timeout 600 models/tt_transformers/tests/test_mlp.py
pytest --timeout 300 models/tt_transformers/tests/test_attention.py
pytest --timeout 300 models/tt_transformers/tests/test_attention_prefill.py
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py
pytest --timeout 400 models/tt_transformers/tests/test_decoder_prefill.py
pytest --timeout 500 models/tt_transformers/tests/test_model.py -k full
"""


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def elf_sections(path):
    data = path.read_bytes()
    assert data[:6] == b"\x7fELF\x01\x01", path
    header = struct.unpack_from("<16sHHIIIIIHHHHHH", data)
    offset, width, count, string_index = header[6], header[11], header[12], header[13]
    headers = [struct.unpack_from("<IIIIIIIIII", data, offset + i * width) for i in range(count)]
    strings = headers[string_index]
    names = data[strings[4] : strings[4] + strings[5]]
    result = []
    for row in headers:
        name = names[row[0] :].split(b"\0", 1)[0].decode()
        if name in (".text", ".data", ".rodata", ".bss"):
            result.append({"name": name, "address": row[3], "bytes": row[5]})
    return result


def capture(start):
    root = Path.home() / ".cache/tt-metal-cache"
    records = []
    for context in root.iterdir() if root.exists() else ():
        router = context / "kernels/fabric_erisc_router"
        if not router.exists():
            continue
        files = list(router.rglob("*"))
        if not any(p.is_file() and p.stat().st_mtime >= start for p in files):
            continue
        files += list((context / "firmware/erisc").rglob("*"))
        args = context / "kernels/kernel_args.csv"
        if args.exists():
            files.append(args)
        for path in files:
            if not path.is_file() or path.suffix not in (
                ".h",
                ".hpp",
                ".cpp",
                ".cc",
                ".log",
                ".csv",
                ".elf",
                ".o",
                ".dephash",
            ):
                continue
            relative = path.relative_to(root)
            target = OUT / "jit" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record = {"path": str(relative), "bytes": path.stat().st_size, "sha256": sha(path)}
            if path.suffix == ".elf":
                record["sections"] = elf_sections(path)
            records.append(record)
    save("jit-receipt.json", records)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == BASE
    forbidden = [k for k in os.environ if k.startswith("TT_METAL_WATCHER_DISABLE_")]
    forbidden += [k for k in ("TT_METAL_WATCHER_NOINLINE", "TT_METAL_WATCHER_APPEND") if k in os.environ]
    assert not forbidden, forbidden
    registry = yaml.safe_load(Path("tests/pipeline_reorg/models_unit_tests.yaml").read_text())
    entry = next(row for row in registry if row["name"] == "Llama 3.1-8B unit tests")
    normalized = (
        "\n".join(line for line in entry["cmd"].splitlines() if line.strip() and not line.lstrip().startswith("#"))
        + "\n"
    )
    assert normalized == RECIPE
    assert entry["skus"]["wh_llmbox_perf"] == {"timeout": 18, "tier": 2}
    wheels = list(Path(".").glob("ttnn-*.whl"))
    assert len(wheels) == 1 and sha(wheels[0]) == WHEEL
    os.environ["TT_METAL_WATCHER"] = "15"
    os.environ["TT_METAL_WATCHER_NOINLINE"] = "1"
    # These two settings add host log output, without changing kernel instrumentation.
    os.environ["TT_METAL_LOG_KERNELS_COMPILE_COMMANDS"] = "1"
    os.environ["TT_LOGGER_LEVEL"] = "info"
    import ttnn

    binding = Path(ttnn._ttnn.__file__)
    assert sha(binding) == BINDING
    compiler = Path("/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++")
    provenance = {
        "diagnostic_only": True,
        "source": BASE,
        "baseline_run": 36752002112,
        "baseline_unit": 110016376210,
        "binding_path": str(binding),
        "binding_sha256": sha(binding),
        "wheel_sha256": sha(wheels[0]),
        "compiler_path": str(compiler),
        "compiler_sha256": sha(compiler),
        "compiler_version": subprocess.check_output([str(compiler), "--version"], text=True),
        "recipe_sha256": hashlib.sha256(RECIPE.encode()).hexdigest(),
        "watcher": {"period": 15, "noinline": 1, "disabled_features": [], "append": False},
        "logging_only": {"TT_METAL_LOG_KERNELS_COMPILE_COMMANDS": "1", "TT_LOGGER_LEVEL": "info"},
        "model_mount": "read-only; exact original model/cache paths",
        "runner": os.environ["RUNNER_NAME"],
    }
    save("provenance.json", provenance)
    (OUT / "original-recipe.sh").write_text(RECIPE)
    print("NOINLINE diagnostic provenance: " + json.dumps(provenance), flush=True)
    start = time.time()
    try:
        result = subprocess.run(["bash", "-e", "-o", "pipefail", "-c", RECIPE])
    finally:
        capture(start)
    save(
        "terminal.json",
        {"returncode": result.returncode, "elapsed_seconds": time.time() - start, "qualification": False},
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
