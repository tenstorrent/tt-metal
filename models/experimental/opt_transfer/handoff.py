import json
from pathlib import Path

_README = """# Claude Code debug handoff

The autonomous bring-up stopped and needs you. Steps:
1. Read `diagnosis_bundle.json` (state + proposals + measured diffs + diagnosis).
2. Use the `debug` skill to root-cause the failing node named in `diagnosis`.
3. Fix the code / KB entry / config, then resume:
   `python -m models.experimental.opt_transfer.run --model {model} --resume`
"""


def dump_bundle(state: dict, run_dir) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnosis_bundle.json").write_text(json.dumps(state, indent=2, default=str))
    (run_dir / "README_FOR_CLAUDE.md").write_text(_README.format(model=state.get("model", "?")))
    return run_dir
