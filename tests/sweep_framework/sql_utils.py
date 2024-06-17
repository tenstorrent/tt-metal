# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


def column_names_to_sql_string(column_names):
    def name_to_string(name):
        if name == "timestamp":
            return "timestamp TIMESTAMP"
        else:
            return f"{name} TEXT"

    column_names = [name_to_string(name) for name in column_names]
    return ", ".join(column_names)
