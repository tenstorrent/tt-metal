# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/sudo_l.sh: sudo probes that run nothing, so every sudo call matches one fixed-argument sudoers line
# (the fixed sudoers list under docs/). 'sudo -n -l <command>' exits 0 only when sudoers allows that exact command; a fixed list
# never allows 'sudo -n true', so a gate on it turns every reader off there. Each probe has a 5 s cap and stdin closed.
# sudo_l <cmd...>: 0 when root, or sudo -n -l allows this exact command
sudo_l() { [ "$(id -u)" = 0 ] || timeout -k 2 5 sudo -n -l "$@" < /dev/null > /dev/null 2>&1; }
# sudo_pfx <cmd...>: prints "sudo -n" when not root and sudo -n -l allows this exact command, else nothing
sudo_pfx() { [ "$(id -u)" != 0 ] && sudo_l "$@" && echo "sudo -n"; return 0; }
# sudo_t <secs> <cmd...>: runs the command under timeout, with sudo -n when sudo_pfx says so, else as is (as with no sudo)
sudo_t() { local t=$1; shift; timeout "$t" $(sudo_pfx "$@") "$@"; }
# sudo_all: root, or sudo -n -l lists NOPASSWD: ALL (for commands whose arguments vary: find over /proc, docker, cat of a stack)
sudo_all() { [ "$(id -u)" = 0 ] || timeout -k 2 5 sudo -n -l < /dev/null 2> /dev/null | grep -qE 'NOPASSWD: *(.*, *)?ALL *(,|$)'; }
