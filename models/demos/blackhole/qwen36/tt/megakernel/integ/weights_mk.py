"""Load-time weight repack for the megakernel (DESIGN_integration.md §4). Placeholder interface: the per-block layout
functions come from the engine tracks (AR track weights.py for MLP pages; v12 split-K; mixed3 attention qkv/wo EXT
slices). ``repack(model)`` returns a dict keyed by (layer, tensor id) once those functions are wired in.
"""


def repack(model):
    raise NotImplementedError(
        "megakernel weight repack: wire in the engine tracks' layout functions (AR weights.py, v12 split-K, mixed3 EXT)"
    )
