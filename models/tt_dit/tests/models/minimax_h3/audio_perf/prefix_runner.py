"""Run the clip comparison against the PRE-FIX build, with hard proof it really used it.

The venv installs ttnn as an *editable* package whose MetaPathFinder hard-maps

    'ttnn' -> /data/rshirvani/tt-metal/ttnn/ttnn

That finder sits on sys.meta_path, which is consulted before sys.path, so PYTHONPATH cannot override
it -- an earlier attempt silently loaded the post-fix .so while claiming to be the pre-fix run. So
repoint the mapping itself, then assert every loaded artefact lives under the prefix tree. Wrong
numbers labelled as a baseline are worse than no numbers.
"""

import importlib
import os
import sys

PREFIX = "/data/rshirvani/tt-metal/.claude/worktrees/h3-prefix"
MAIN = "/data/rshirvani/tt-metal"

# 1. repoint the editable finder's mapping at the prefix tree
finder_name = next(
    (
        m
        for m in os.listdir(os.path.join(sys.prefix, "lib", "python3.10", "site-packages"))
        if m.startswith("__editable___ttnn") and m.endswith("_finder.py")
    ),
    None,
)
assert finder_name, "editable ttnn finder not found"
finder = importlib.import_module(finder_name[:-3])
finder.MAPPING = {k: v.replace(MAIN, PREFIX, 1) for k, v in finder.MAPPING.items()}
print("remapped finder:", finder.MAPPING)

# 2. make the prefix tree win on sys.path too (for `models.*`), and drop the main checkout
sys.path[:] = [p for p in sys.path if p.rstrip("/") not in (MAIN, f"{MAIN}/ttnn", f"{MAIN}/tools")]
sys.path.insert(0, f"{PREFIX}/ttnn")
sys.path.insert(0, PREFIX)

# 3. verify, loudly
import ttnn  # noqa: E402

print("ttnn      ->", ttnn.__file__)
assert ttnn.__file__.startswith(PREFIX), f"ttnn came from {ttnn.__file__}, not the prefix build"

import ttnn._ttnn as _t  # noqa: E402

print("_ttnn.so  ->", _t.__file__)
assert _t.__file__.startswith(PREFIX), f"_ttnn.so came from {_t.__file__}"

import models.tt_dit.layers.audio_ops as ao  # noqa: E402

print("audio_ops ->", ao.__file__)
assert ao.__file__.startswith(PREFIX), f"model code came from {ao.__file__}"
# the pre-fix tree must NOT contain the post-fix padding fix
src = open(ao.__file__).read()
assert "ttnn.pad(x_BTC" not in src, "prefix tree unexpectedly contains the post-fix padding change"
print("confirmed: pre-fix model code (no ttnn.pad channel fix)")

# 4. run the comparison
sys.argv = [sys.argv[0]]
runpy_target = f"{PREFIX}/models/tt_dit/tests/models/minimax_h3/audio_perf/cpu_vs_device.py"
code = compile(open(runpy_target).read(), runpy_target, "exec")
g = {"__name__": "__main__", "__file__": runpy_target}
exec(code, g)
