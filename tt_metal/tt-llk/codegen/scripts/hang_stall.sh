# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Default log-stall threshold (seconds) for run_test.sh's hang watcher. Sourced.
#
#   llk_default_stall MODE     MODE = simulator | hardware; prints the seconds.
#
# HANG_STALL wins. With ENABLE_WAVEFORM on, the emulator records the whole
# design and a healthy run can print nothing for many minutes, so the emulator
# threshold rises to WAVE_HANG_STALL (default 1800).

llk_default_stall() {
  if [[ -n "${HANG_STALL:-}" ]]; then
    echo "$HANG_STALL"
  elif [[ "$1" == "simulator" ]]; then
    case "${ENABLE_WAVEFORM:-0}" in
      1|true|TRUE|yes) echo "${WAVE_HANG_STALL:-1800}" ;;
      *) echo 180 ;;
    esac
  else
    echo 300
  fi
}
