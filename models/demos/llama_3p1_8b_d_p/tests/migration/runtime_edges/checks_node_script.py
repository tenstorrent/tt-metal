"""Run shell path binding with a harmless stdlib recorder instead of an owner."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NodeScriptTests(unittest.TestCase):
    # A path with spaces must preserve exact plan/hash arguments and force the accepted native environment.
    def test_actual_shell_uses_hashed_environment_and_supervisor_arguments(self):
        with tempfile.TemporaryDirectory(prefix="publication paths ") as tmp:
            root = Path(tmp)
            out = root / "record.json"
            recorder = root / "record-python"
            recorder.write_text(
                "#!"
                + sys.executable
                + '\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["RECORD_PATH"]).write_text(json.dumps({"argv":sys.argv[1:],"fabric":os.environ.get("PREFILL_FABRIC_MODE"),"slow":os.environ.get("TT_METAL_SLOW_DISPATCH_MODE"),"threads":os.environ.get("OMP_NUM_THREADS")}))\n'
            )
            recorder.chmod(0o700)
            envfile = root / "environment.sh"
            envfile.write_text(
                "export PREFILL_REPO="
                + repr(str(root))
                + "\nexport PREFILL_PYTHON="
                + repr(str(recorder))
                + "\nexport TT_METAL_SLOW_DISPATCH_MODE=1 OMP_NUM_THREADS=9 PREFILL_FABRIC_MODE=bad\n"
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(dict(environment_script=str(envfile), source_pins={str(envfile): digest(envfile)}))
            )
            env = dict(os.environ, RECORD_PATH=str(out))
            argv = ["bash", str(HERE / "node-run.sh"), str(plan), digest(plan)]
            done = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=10)
            self.assertEqual(done.returncode, 0, done.stderr)
            row = json.loads(out.read_text())
            self.assertEqual(
                row["argv"],
                ["-B", str(HERE / "node-supervisor.py"), "--plan", str(plan), "--plan-sha256", digest(plan)],
            )
            self.assertEqual((row["fabric"], row["slow"], row["threads"]), ("1d_ring", None, "1"))
            out.unlink()
            envfile.write_text(envfile.read_text() + "\n# changed\n")
            failed = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
