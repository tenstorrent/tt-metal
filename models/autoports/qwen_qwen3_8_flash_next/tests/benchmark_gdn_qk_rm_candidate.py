# SPDX-License-Identifier: Apache-2.0
"""Isolated ROW_MAJOR normalized Q/K reader; no production module mutation."""

import ast
import inspect
import textwrap

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as decoder


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one original reader fragment: {old}")
    return source.replace(old, new)


def make_reader(source):
    source = replace_once(
        source,
        """            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {""",
        """            if(which!=3) {
                const uint32_t row=(b*length+token)*4+h/3;
                if(which==1) noc.async_read(q,CoreLocalMem<uint32_t>(scratch),512,{.page_id=row},{});
                else noc.async_read(k,CoreLocalMem<uint32_t>(scratch),512,{.page_id=row},{});
                noc.async_read_barrier();
            }
            for (uint32_t tile=0;tile<(which==3?1:4);++tile) {""",
    )
    for which, name in ((1, "q"), (2, "k")):
        source = replace_once(
            source,
            f"                if (which=={which}) noc.async_read({name},CoreLocalMem<uint32_t>(scratch),4096,{{.page_id=page}},{{}});\n",
            "",
        )
    return replace_once(source, "const uint32_t value=src[off(h/3,r)];", "const uint32_t value=src[tile*32+r];")


READER = make_reader(decoder._BROADCAST_PREFILL_READER)


def normalized_rm(q, k):
    result = []
    for value in (q, k):
        if value.dtype != ttnn.float32 or tuple(value.shape)[2:] != (4, 128):
            raise ValueError("Expected normalized FP32 [B,S,4,128]")
        converted = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=value.memory_config())
        assert converted.dtype == value.dtype and tuple(converted.shape) == tuple(value.shape)
        result.append(converted)
    return tuple(result)


def make_recurrence():
    """Clone only the Python launcher namespace, retaining its original CBs."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(decoder._local_recurrence)))
    function = tree.body[0]
    anchor = next(
        i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
    )
    function.body[anchor + 1 : anchor + 1] = ast.parse(
        "assert self.recurrence_broadcast and batch == 1 and length in (128,256,512)\nq, k = normalized_rm(q, k)"
    ).body
    ast.fix_missing_locations(tree)
    namespace = dict(vars(decoder), normalized_rm=normalized_rm, _BROADCAST_PREFILL_READER=READER)
    exec(compile(tree, __file__, "exec"), namespace)
    return namespace["_local_recurrence"]


RECURRENCE = make_recurrence()


def make_gdn_recurrence(local_recurrence=RECURRENCE):
    """Keep all production normalization operations exactly as written."""
    source = textwrap.dedent(inspect.getsource(decoder.ParallelGatedDeltaNet._recurrence))
    namespace = dict(vars(decoder), _local_recurrence=local_recurrence)
    exec(compile(source, __file__, "exec"), namespace)
    return namespace["_recurrence"]


GDN_RECURRENCE = make_gdn_recurrence()
