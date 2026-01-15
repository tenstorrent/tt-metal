import importlib.metadata
import traceback

TENSTORRENT_PROVIDER = {}


# Tenstorrent ttnn library
try:
    TENSTORRENT_PROVIDER["ttnn"] = {"ttnn": importlib.metadata.version("ttnn")}
except:
    pass

# Tenstorrent tt-metal library (legacy)
try:
    TENSTORRENT_PROVIDER["tt_lib"] = {"tt_lib": importlib.metadata.version("tt-lib")}
except:
    try:
        # Alternative package name
        TENSTORRENT_PROVIDER["tt_lib"] = {"tt_lib": importlib.metadata.version("tt_metal")}
    except:
        pass


# Tenstorrent-optimized vLLM
try:
    TENSTORRENT_PROVIDER["vllm"] = {
        "vllm": importlib.metadata.version("vllm"),
    }
except:
    pass


# PyTorch (always required)
try:
    TENSTORRENT_PROVIDER["torch"] = {"torch": importlib.metadata.version("torch")}
except:
    pass
