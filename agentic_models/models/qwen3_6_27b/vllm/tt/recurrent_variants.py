# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact source edits for isolated local12 recurrence experiments (no TTNN import).

Candidates are restricted by their caller to raw packed QKV. Neither arithmetic,
compile-time accessor offsets, CB descriptors, nor the writer are changed.
"""

ATOMIC_VARIANTS = ("skip_dead_padding", "skip_constants", "grouped_reads", "gate_cache", "exp_row", "exp_scalar")
CUMULATIVE = ("skip_dead_padding", "skip_constants", "grouped_reads", "gate_cache")
VARIANTS = ("baseline", *ATOMIC_VARIANTS, "cumulative", "cumulative_exp_row", "cumulative_exp_scalar")


def _replace(source, before, after):
    count = source.count(before)
    if count != 1:
        raise AssertionError(f"Expected one source anchor, found {count}: {before!r}")
    return source.replace(before, after, 1)


def sources(reader, compute, variant="baseline", *, grouped=True, poison_dead_padding=False):
    """Return reader/compute source; reject drift instead of silently partial edits.

    `grouped` must equal the COMPUTE compile-time GROUP_QK value after the host
    partition resolves it. It is a reader literal, leaving accessor base 8 intact.
    `poison_dead_padding` is an expensive correctness-only NaN-padding control;
    never use its timings as candidate performance evidence.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown recurrent variant {variant!r}; choose {VARIANTS}")
    selected = set(CUMULATIVE if variant.startswith("cumulative") else (variant,))
    if variant.startswith("cumulative_exp_"):
        selected.add(variant.removeprefix("cumulative_"))
    if "skip_dead_padding" in selected:
        reader = _replace(
            reader,
            "cb.reserve_back(n); noc.async_write_zeros(cb,2*n*4096);",
            "cb.reserve_back(n); if(id<2) noc.async_write_zeros(cb,2*n*4096);",
        )
    if poison_dead_padding:
        reader = _replace(
            reader,
            "    noc.write_zeros_l1_barrier();",
            """    noc.write_zeros_l1_barrier();
    // Correctness control: poison both V/g/beta slots; live stores overwrite them.
    for(uint32_t id=2;id<5;id++) {
        CircularBuffer cb(id); const uint32_t n=id==2?4:1;
        volatile tt_l1_ptr uint32_t* p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for(uint32_t i=0;i<2*n*1024;i++) p[i]=0x7fc00000u;
    }""",
        )
    if "skip_constants" in selected:
        reader = _replace(
            reader,
            """    for(uint32_t id=6;id<=7;id++) {
        CircularBuffer cb(id); cb.reserve_back(1);
        volatile tt_l1_ptr uint32_t* p=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for(uint32_t i=0;i<1024;i++) p[i]=id==7?0x3f800000u:0u;
        cb.push_back(1);
    }
""",
            "",
        )
        compute = _replace(compute, "    zero.wait_front(1); one.wait_front(1);\n", "")
        compute = _replace(compute, "    zero.pop_front(1); one.pop_front(1);\n", "")
    if "grouped_reads" in selected:
        reader = _replace(
            reader,
            "    constexpr auto qa=TensorAccessorArgs<8>();",
            f"    constexpr bool GROUP_QK={'true' if grouped else 'false'};\n"
            "    constexpr auto qa=TensorAccessorArgs<8>();",
        )
        reader = _replace(
            reader,
            """        const uint32_t user=h/12,hv=h%12,hk=hv/3;
        if(h!=start) for(uint32_t id=0;id<5;id++) {
            CircularBuffer cb(id); cb.reserve_back(id<3?4:1);
""",
            """        const uint32_t user=h/12,hv=h%12,hk=hv/3;
        const bool prepare_qk=!GROUP_QK || h==start || h%3==0;
        if(h!=start) for(uint32_t id=0;id<5;id++) {
            if(id<2 && !prepare_qk) continue;
            CircularBuffer cb(id); cb.reserve_back(id<3?4:1);
""",
        )
        for line in (
            "            noc.async_read(q,scratch,256,{.page_id=user,.offset_bytes=hk*256},{.offset_bytes=0});",
            "            noc.async_read(k,scratch,256,{.page_id=user,.offset_bytes=(PACKED?1024:0)+hk*256},{.offset_bytes=512});",
            "            faces(q,NORMALIZED?user*4:user*Q_STRIDE+hk*4,NORMALIZED?hk:0,NORMALIZED?4:2,0);",
            "            faces(k,NORMALIZED?user*4:user*Q_STRIDE+hk*4,NORMALIZED?hk:0,NORMALIZED?4:2,512);",
        ):
            reader = _replace(reader, line, line.replace("            ", "            if(prepare_qk) ", 1))
        reader = _replace(
            reader,
            """        for(uint32_t id=0;id<3;id++) {
            CircularBuffer cb(id);""",
            """        for(uint32_t id=0;id<3;id++) {
            if(id<2 && !prepare_qk) continue;
            CircularBuffer cb(id);""",
        )
        compute = _replace(
            compute,
            """        } else {
            // Reader's rings remain one head deep: consume every duplicate input
            // to permit it to fetch the next state/value/gate without stalling.
            raw_q.wait_front(4); raw_k.wait_front(4);
            raw_q.pop_front(4); raw_k.pop_front(4);
        }
""",
            "        }\n",
        )
    if "gate_cache" in selected:
        reader = _replace(
            reader,
            "        gate(g,user,hv,G_FP32?4:2,1536); gate(b,user,hv,B_FP32?4:2,1600);",
            """        if(h==start || h%12==0) {
            gate(g,user,hv,G_FP32?4:2,1536); gate(b,user,hv,B_FP32?4:2,1600);
        }""",
        )
    if "exp_row" in selected or "exp_scalar" in selected:
        replacement = (
            "exp_tile<false>(0,VectorMode::R)"
            if "exp_row" in selected
            else "exp_tile<false,false,InputClamping::ClampToNegative,1>(0,VectorMode::None)"
        )
        compute = _replace(
            compute,
            "exp_tile_init<false>(); exp_tile<false>(0); finish(18,0);",
            f"exp_tile_init<false>(); {replacement}; finish(18,0);",
        )
    return reader, compute
