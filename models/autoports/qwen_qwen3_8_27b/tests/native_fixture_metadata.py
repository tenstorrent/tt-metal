# SPDX-License-Identifier: Apache-2.0
"""Record actual linear kernel arguments during untimed native preparation."""
import contextlib
import json


@contextlib.contextmanager
def capture_linear_calls(record, ttnn, model):
    owners = {id(model.head): "head"}
    for layer in model.layers:
        for role in ("mix", "out", "gate", "up", "gateup", "down"):
            weight = getattr(layer, role, None)
            if weight is not None:
                owners[id(weight)] = f"layers.{layer.layer_idx}.{role}"
    observed = {}
    original = ttnn.linear

    def linear(*args, **kwargs):
        config = kwargs.get("compute_kernel_config")
        if len(args) >= 2 and id(args[1]) in owners and config is not None:
            program = kwargs.get("program_config")
            row = dict(
                owner=owners[id(args[1])],
                input_dtype=str(args[0].dtype),
                weight_dtype=str(args[1].dtype),
                input_shape=list(args[0].shape),
                weight_shape=list(args[1].shape),
                math_fidelity=str(config.math_fidelity),
                fp32_dest_acc_en=config.fp32_dest_acc_en,
                program_class=type(program).__name__ if program is not None else None,
                program={
                    name: getattr(program, name, None)
                    for name in ("in0_block_w", "per_core_M", "per_core_N", "num_workers_per_dram_bank")
                },
            )
            key = json.dumps(row, sort_keys=True)
            if key not in observed:
                observed[key] = dict(row, calls=0)
            observed[key]["calls"] += 1
        return original(*args, **kwargs)

    ttnn.linear = linear
    try:
        yield
    finally:
        ttnn.linear = original
        record["prepared_linear_calls"] = list(observed.values())
