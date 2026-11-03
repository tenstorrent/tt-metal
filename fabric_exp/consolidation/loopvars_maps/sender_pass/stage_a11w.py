#!/usr/bin/env python3
"""stage_a11w.py NAME [--read-group] [--shape]: a study variant off v4txa11 (user go 2026-11-03; v4txa11 untouched) that
changes only PartialPresetForwards::forward (router_downstream.hpp), the forward of multi-downstream receivers:
  --read-group  the forward's two command-buffer ready reads and the `node` load inside begin_read_group() /
                end_read_group() (the documented grouping primitive), instead of relying on the scheduler (v4txa11g alone)
  --shape       the decoupled slow path (v4txa11w): fast path (both buffers ready) as today, trid + data command, then
                the credit command; slow path waits for the data buffer, issues the data command, then waits for the
                credit buffer and issues the credit command (today's slow path waits for both before either). The two
                commands are written once (data_command, credit_command) and called from both paths.
Check: E0 and every single-downstream E1 image identical to v4txa11; only multi-downstream E1 images change."""
import os
import shutil
import sys

V = "/home/aagarwal/tt-metal/fabric_exp/variants"
name = sys.argv[1]
rg, shape = "--read-group" in sys.argv, "--shape" in sys.argv
assert rg or shape
src, dst = f"{V}/v4txa11", f"{V}/{name}"
if os.path.exists(dst):
    sys.exit(f"{dst} exists")
shutil.copytree(src, dst, symlinks=True)
p = f"{dst}/tt_metal/fabric/hw/inc/edm_fabric/router/router_downstream.hpp"
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


READS = '''        // Read once: the register writes below store through uint32_t pointers, which may alias the field.
        const uint32_t node = link.noc_coord;
        const bool data_ready = noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf);
        const bool credit_ready = noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf);
'''
READS_RG = '''        // Read once (the register writes below store through uint32_t pointers, which may alias the field), with
        // the two buffers' ready state, as one read group.
        begin_read_group();
        const uint32_t node = link.noc_coord;
        const bool data_ready = noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf);
        const bool credit_ready = noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf);
        end_read_group();
'''
TAIL = '''        if (!(data_ready & credit_ready)) {
            while (!noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf));
            while (!noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf));
        }
        ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, trid);
        noc_write_to_node_with_state<
            DM_DEDICATED_NOC,
            Traits::data_cmd_buf,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_wait,
            /*update_counter=*/is_blackhole_arch,
            /*posted=*/true>(Traits::noc, src_addr, node, dst, size_bytes);
        noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
            node,
            get_stream_reg_write_addr<Traits::template remote_credit_stream<ENTRY>>(),
            Traits::sync_cmd_buf,
            Traits::noc);
    }

private:
'''
TAIL_SHAPE = '''        if (data_ready & credit_ready) {
            // Fast path: both buffers ready, the two commands back to back.
            data_command(node, dst, src_addr, size_bytes, trid);
            credit_command<ENTRY>(node);
        } else {
            // Slow path: each command waits only for its own buffer, so the data command does not wait behind the
            // previous forward's credit (two forwards in a row: a multicast fanning out both ways).
            while (!noc_cmd_buf_ready(Traits::noc, Traits::data_cmd_buf));
            data_command(node, dst, src_addr, size_bytes, trid);
            while (!noc_cmd_buf_ready(Traits::noc, Traits::sync_cmd_buf));
            credit_command<ENTRY>(node);
        }
    }

private:
    // The forward's data command: its transaction ID, then the packet into `dst` on `node`.
    FORCE_INLINE static void data_command(
        uint32_t node, uint32_t dst, uint32_t src_addr, uint32_t size_bytes, uint8_t trid) {
        ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(Traits::noc, Traits::data_cmd_buf, trid);
        noc_write_to_node_with_state<
            DM_DEDICATED_NOC,
            Traits::data_cmd_buf,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_wait,
            /*update_counter=*/is_blackhole_arch,
            /*posted=*/true>(Traits::noc, src_addr, node, dst, size_bytes);
    }

    // The forward's credit command: -1 on the free-slots register of downstream entry ENTRY on `node`.
    template <size_t ENTRY>
    FORCE_INLINE static void credit_command(uint32_t node) {
        noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
            node,
            get_stream_reg_write_addr<Traits::template remote_credit_stream<ENTRY>>(),
            Traits::sync_cmd_buf,
            Traits::noc);
    }

'''
DOC = '''    // The packet into `dst` on the node at `link`, tagged with `trid`, then -1 on that node's free-slots register. The
    // two buffers' ready state is read as a pair, so the two reads overlap, and tested together: when
    // either is busy (rare), both are waited for before either command, so no ready state is held across the data
    // command. Only this ERISC issues on them, so a buffer seen ready stays ready until its next command.
'''
DOC_SHAPE = '''    // The packet into `dst` on the node at `link`, tagged with `trid`, then -1 on that node's free-slots register. The
    // two buffers' ready state is read as a pair, so the two reads overlap, and tested together: both ready, the two
    // commands go back to back; otherwise each command waits for its own buffer just before it. Only this ERISC issues
    // on them, so a buffer seen ready stays ready until its next command.
'''
if rg:
    rep(READS, READS_RG)
if shape:
    rep(DOC, DOC_SHAPE)
    rep(TAIL, TAIL_SHAPE)
open(p, "w").write(s)
print(f"wrote {dst} ({'read group' if rg else ''}{' + ' if rg and shape else ''}{'shape' if shape else ''})")
