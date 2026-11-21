"""One way to get the result tensor out of whatever wrapper a module returned it in.

Several places need this and each had grown its own version: a nested closure in `e2e_harness`, a
richer ttnn-aware converter in `activation_diff`, and two copies inside the generated-test
templates. They agreed on the easy cases and diverged on the interesting ones, which is the usual
result of a pattern living in four heads at once.

The wrapper is asked to unpack itself rather than being reached into by field name. A HuggingFace
output orders its own fields and omits the empty ones, so the first tensor out of `to_tuple()` is
the result whether the model calls it logits, hidden states, or something this has never heard of
-- and a model that renames its output tomorrow still works, because no name was ever typed here.
"""

from __future__ import annotations

from typing import Any, Optional


def result_tensor(out: Any) -> Optional[Any]:
    """The tensor `out` carries, or None if it carries none.

    Recurses through sequences and mappings, so a `(logits, past_key_values)` tuple or a dict of
    outputs resolves to the tensor rather than to the container.
    """
    import torch

    if isinstance(out, torch.Tensor):
        return out
    unpack = getattr(out, "to_tuple", None)
    if callable(unpack):
        try:
            out = unpack()
        except Exception:  # noqa: BLE001 -- a wrapper that will not unpack is treated as opaque
            return None
    elif isinstance(out, dict):
        out = list(out.values())
    if isinstance(out, (list, tuple)):
        for item in out:
            found = result_tensor(item)
            if found is not None:
                return found
    return None
