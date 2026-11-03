"""probes.py: v4rxnsbr and v4rxnsbs = v4rxnsb with one change each to forward_partially_preset's ready handling
(V4_NS_PRESETS §8.3): r = the lead's pattern (each buffer waited for just before its command; reads still paired),
s = serial polls (each command polls its own buffer)."""
import os, shutil, sys
V = "/home/aagarwal/tt-metal/fabric_exp/variants"
R = "tt_metal/fabric/hw/inc/edm_fabric/router/router_downstream.hpp"
JOINT = """    const bool data_ready = noc_cmd_buf_ready(NOC, DATA_CMD_BUF);
    const bool credit_ready = noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF);
    if (!(data_ready & credit_ready)) {
        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
    }
    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(NOC, DATA_CMD_BUF, trid);"""
CREDIT = """    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
        node, credit_register, CREDIT_CMD_BUF, NOC);"""
DOC_JOINT = """// pair, so the two reads overlap (RESULTS §17), and tested together: when either is busy (rare), both are waited for
// before either command, so no ready state is held across the data command. Only this ERISC issues on them, so a buffer
// seen ready stays ready until its next command."""
NSBS_DOC = """// A forward on the partially preset buffers (forwards_use_presets): the packet into `dst` on the node at `link`, tagged
// with `trid`, then -1 on that node's free-slots register `credit_register`. Each command polls its own buffer just
// before it (PROBE nsbs, V4_NS_PRESETS §8.3: serial polls; no ready state is read ahead or held).
"""
FORMS = {
    "v4rxnsbr": (
        """    const bool data_ready = noc_cmd_buf_ready(NOC, DATA_CMD_BUF);
    const bool credit_ready = noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF);
    if (!data_ready) {
        while (!noc_cmd_buf_ready(NOC, DATA_CMD_BUF));
    }
    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/true>(NOC, DATA_CMD_BUF, trid);""",
        """    if (!credit_ready) {
        while (!noc_cmd_buf_ready(NOC, CREDIT_CMD_BUF));
    }
    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/true>(
        node, credit_register, CREDIT_CMD_BUF, NOC);""",
        """// pair, so the two reads overlap (RESULTS §17); a buffer not ready yet is waited for just before its command (PROBE
// nsbr, V4_NS_PRESETS §8.3: the lead's pattern). Only this ERISC issues on them, so a buffer seen ready stays ready
// until its next command."""),
    "v4rxnsbs": (
        """    ncrisc_noc_set_transaction_id</*skip_cmdbuf_chk=*/false>(NOC, DATA_CMD_BUF, trid);""",
        """    noc_inline_dw_write_to_node_with_state</*update_counter=*/false, /*posted=*/true, /*skip_cmdbuf_chk=*/false>(
        node, credit_register, CREDIT_CMD_BUF, NOC);""",
        """// pair... (PROBE nsbs, V4_NS_PRESETS §8.3: serial polls instead: each command polls its own buffer just before it, so
// no ready state is read ahead or held). Only this ERISC issues on them, so a buffer seen ready stays ready
// until its next command."""),
}
for name, (head, credit, doc) in FORMS.items():
    dst = f"{V}/{name}"
    if os.path.exists(dst):
        sys.exit(f"{dst} exists")
    shutil.copytree(f"{V}/v4rxnsb", dst, symlinks=True)
    p = f"{dst}/{R}"
    s = open(p).read()
    for o, n in ((JOINT, head), (CREDIT, credit), (DOC_JOINT, doc)):
        assert s.count(o) == 1, (name, o[:50])
        s = s.replace(o, n)
    if name == "v4rxnsbs":  # the whole doc paragraph: no paired reads in this form
        a = s.index("// A forward on the partially preset buffers (forwards_use_presets)")
        b = s.index("template <uint8_t NOC, uint8_t DATA_CMD_BUF, uint8_t CREDIT_CMD_BUF>\nFORCE_INLINE void forward_partially_preset(", a)
        s = s[:a] + NSBS_DOC + s[b:]
    open(p, "w").write(s)
    print("made", dst)
