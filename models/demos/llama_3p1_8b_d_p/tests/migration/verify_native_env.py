import json
import os
import sys
from pathlib import Path

import pytest
import torch
import transformers

import ttnn

repo = Path(os.environ["PREFILL_REPO"]).resolve()
binding = Path(ttnn._ttnn.__file__).resolve()
assert binding.is_relative_to(repo), (binding, repo)
libs = sorted(
    {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/" in line and any(n in line for n in ("libtt_metal", "libttnn", "libtt-umd", "libtt_stl", "_ttnn"))
    }
)
assert libs and all(Path(p).resolve().is_relative_to(repo) for p in libs), libs
config_paths = {
    n: str(getattr(ttnn.CONFIG, n)) for n in ("cache_path", "model_cache_path", "tmp_dir", "root_report_path")
}
assert all(Path(v).is_relative_to(repo.parent.parent) for v in config_paths.values()), config_paths
print(
    json.dumps(
        dict(
            config_paths=config_paths,
            python=sys.version,
            torch=torch.__version__,
            transformers=transformers.__version__,
            pytest=pytest.__version__,
            ttnn=ttnn.__file__,
            binding=str(binding),
            native_libraries=libs,
            checkpoint_config=(Path(os.environ["HF_MODEL"]) / "config.json").is_file(),
        ),
        indent=2,
    )
)
