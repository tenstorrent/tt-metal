# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only CSV round trips; runnable directly without pytest/TTNN fixtures."""

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CsvExportQuotingTest(unittest.TestCase):
    def test_zone_fields_round_trip_without_losing_rows_or_source(self):
        root = Path(__file__).resolve().parents[3]
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "A C++ compiler is required for the actual exporter helper")
        sources = [
            "kernel.cpp",
            "compute.cpp,\nreader.cpp,\nwriter.cpp",
            '#include "dataflow_api.h"\nconstexpr int values[] = {1, 2, 3};\n',
            'quoted "path", with commas\r\nand a final newline\n',
            "semicolon;path",
            "",
        ]
        rows = [
            ["Program_11980", source, "0", "", 'id:1, text "quoted"', "123", "456", "39168", "id:1"]
            for source in sources
        ]
        # Include every separator in one round trip, empty trailing fields and
        # an actual TT host zone_text which must not be double-quoted as data.
        rows.append(["TT_DNN_DEVICE_OP", "source.cpp", "246", "", "id:1195008", "109", "49816", "1", ""])
        with tempfile.TemporaryDirectory(prefix="tracy-csv-quoting-") as tmp:
            tmp = Path(tmp)
            source = tmp / "round_trip.cpp"
            binary = tmp / "round_trip"
            initializer = ",\n".join("{" + ",".join(json.dumps(field) for field in row) + "}" for row in rows)
            source.write_text(
                '#include "tt_metal/third_party/tracy/csvexport/src/CsvWriter.hpp"\n'
                "#include <iostream>\n#include <vector>\n"
                "int main(int argc, char** argv) {\n"
                "std::vector<std::vector<std::string>> rows = {" + initializer + "};\n"
                'for (const auto& row : rows) std::cout << join_csv(row, argv[1]) << "\\n";\n}\n'
            )
            subprocess.run(
                [compiler, "-std=c++20", "-Wall", "-Wextra", "-I", str(root), str(source), "-o", str(binary)],
                check=True,
            )
            for separator in (",", ";", "\t"):
                with self.subTest(separator=separator):
                    output = subprocess.check_output([str(binary), separator])
                    recovered = list(
                        csv.reader(io.StringIO(output.decode(), newline=""), delimiter=separator, strict=True)
                    )
                    self.assertEqual(recovered, rows)
                    self.assertTrue(all(len(row) == 9 for row in recovered))
                    # Also verify the pandas engine used as the installed
                    # profiler's fallback accepts quoted multiline records.
                    import pandas as pd

                    frame = pd.read_csv(
                        io.BytesIO(output), sep=separator, header=None, dtype=str, keep_default_na=False
                    )
                    self.assertEqual(frame.values.tolist(), rows)


if __name__ == "__main__":
    unittest.main()
