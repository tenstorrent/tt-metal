#!/usr/bin/env python3
"""Install/remove a synthetic 2D receiver forwarding stall. No hardware access.
Run from tt-metal root. Default: show proposed diff only. --apply modifies one
kernel source. --undo removes only the exact inserted block, preserving other edits.
"""
import argparse
import difflib
from pathlib import Path
import re
import sys

SOURCE = Path('tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp')
BEGIN = '// BEGIN TEST_ONLY_2D_RECEIVER_STALL'
END = '// END TEST_ONLY_2D_RECEIVER_STALL'
BLOCK = '''// BEGIN TEST_ONLY_2D_RECEIVER_STALL
#if defined(FABRIC_2D)
// Synthetic fault: keep 1008-byte payload packets resident in RX buffers.
// Keep running the router loop, ACK handling, and normal buffer-credit checks.
// The payload size is the sole discriminator: use only on a reserved test system.
if (packet_header->payload_size_bytes == 1008u) {
    can_send_to_all_local_chip_receivers = false;
}
#endif
// END TEST_ONLY_2D_RECEIVER_STALL
'''

def transform(text, undo=False):
    if undo:
        pattern = r'(?m)^(?P<indent>[ \t]*)' + re.escape(BEGIN) + r'\n[\s\S]*?^[ \t]*' + re.escape(END) + r'\n'
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError('Expected exactly one installed fault block; no changes made.')
        m = matches[0]
        expected = ''.join(m['indent'] + line + '\n' for line in BLOCK.splitlines())
        if m[0] != expected:
            raise ValueError('Fault block has been edited; refusing to remove it automatically.')
        return text[:m.start()] + text[m.end():]
    if BEGIN in text:
        raise ValueError('Fault already installed. Use --undo to remove it.')
    pattern = r'(?m)^(?P<indent>[ \t]*)if\s*\(can_send_to_all_local_chip_receivers\)\s*\{'
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError('Expected exactly one receiver-forwarding gate; source version differs. No changes made.')
    m = matches[0]
    if 'packet_header' not in text[max(0,m.start()-10000):m.start()]:
        raise ValueError('Receiver packet-header scope not found. No changes made.')
    block = ''.join(m['indent'] + line + '\n' for line in BLOCK.splitlines())
    return text[:m.start()] + block + text[m.start():]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    g = p.add_mutually_exclusive_group()
    g.add_argument('--apply', action='store_true')
    g.add_argument('--undo', action='store_true')
    args = p.parse_args()
    path = args.repo.resolve() / SOURCE
    before = path.read_text()
    after = transform(before, args.undo)
    sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/'+str(SOURCE), tofile='b/'+str(SOURCE)))
    if args.apply or args.undo:
        path.write_text(after)
        print('Source updated. Force device JIT recompilation on the next launch.',file=sys.stderr)
    else:
        print('Preview only; no files changed. Use --apply to install.',file=sys.stderr)
if __name__ == '__main__':
    try: main()
    except (ValueError,OSError) as exc: sys.exit(str(exc))
