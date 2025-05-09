# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

_format_log = []


def add_to_format_log(input_fmt, output_fmt):
    global _format_log
    _format_log.append((input_fmt, output_fmt))
