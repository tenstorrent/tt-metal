"""Diagnostic-only fail-closed cache-write guard; same gate in both source legs."""
import json
import os
from pathlib import Path
import sys

if os.environ.get("FORMATTER_CACHE_GUARD") == "1":
    try:
        import ttnn

        def prohibit_tensor_cache_dump(*args, **kwargs):
            Path(os.environ["FORMATTER_CACHE_GUARD_SENTINEL"]).touch()
            raise RuntimeError("Diagnostic prohibits tensor-cache generation or regeneration")

        ttnn._ttnn.tensor.dump_tensor_flatbuffer = prohibit_tensor_cache_dump
        assert ttnn._ttnn.tensor.dump_tensor_flatbuffer is prohibit_tensor_cache_dump
        with Path(os.environ["FORMATTER_CACHE_GUARD_RECEIPTS"]).open("a") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "guard": "tensor-cache-dump-blocked"}) + "\n")
    except BaseException as error:
        # Python normally ignores sitecustomize errors. This gate must fail closed.
        print("Diagnostic cache guard could not be installed: " + type(error).__name__, file=sys.stderr, flush=True)
        os._exit(96)

# Read-only profile observation is required in all three stock boots.
if os.environ.get("FORMATTER_PROFILE_RECORD"):
    try:
        import ttnn
        import producer_profile

        producer_profile.install_observer(
            os.environ["FORMATTER_PROFILE_RECORD"],
            os.environ["FORMATTER_PROFILE_SOURCE"],
            os.environ["FORMATTER_PROFILE_PHASE"],
        )
        if os.environ.get("FORMATTER_CACHE_PRODUCER") == "1":
            assert os.environ.get("FORMATTER_CACHE_GUARD") != "1"
            root = Path("/task-cache/meta-llama--Llama-3.1-8B-Instruct/T3K")
            root.mkdir(parents=True, exist_ok=True)
            ttnn._ttnn.tensor.dump_tensor_flatbuffer = producer_profile.cap_dump(
                ttnn._ttnn.tensor.dump_tensor_flatbuffer, root, Path(os.environ["FORMATTER_CACHE_CAP_RECEIPTS"])
            )
    except BaseException as error:
        print("Diagnostic profile/cap could not be installed: " + type(error).__name__, file=sys.stderr, flush=True)
        os._exit(96)
