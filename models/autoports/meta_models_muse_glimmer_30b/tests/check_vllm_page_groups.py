# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only ownership checks for shared persistent native-group page tables."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from check_vllm_sampling_state import ADAPTER, GENERATOR, extracted_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    namespace = {"torch": torch}
    adapter_type = extracted_class(
        ADAPTER, "MuseGlimmerForConditionalGeneration", {"_tables", "configure_kv_cache_groups"}, namespace
    )
    generator_type = extracted_class(GENERATOR, "MuseGlimmerGenerator", {"_refresh_table"}, namespace)
    gen = generator_type()
    gen.per_layer_page_tables = True
    gen.batch_size = 4
    gen.num_pages = 4096
    gen._page_table_representatives = [0, 1, 0, 1]
    gen._layer_page_hosts = [None] * 4
    gen.layer_page_tables = [torch.zeros((4, 4096), dtype=torch.int32) for _ in range(2)] * 2
    gen.counters = Counter()

    def write(value, tensor, counter):
        tensor.copy_(value)
        gen.counters[counter] += 1

    gen._write = write
    adapter = adapter_type()
    adapter.batch_size = 4
    adapter.model = SimpleNamespace(layer_indices=[0, 1, 2, 3], config=SimpleNamespace(layer_types=[None] * 4))
    adapter.generator = None
    adapter.configure_kv_cache_groups([9, 2, 9, 2])
    assert adapter.page_table_groups == [9, 2, 9, 2]
    adapter.generator = gen
    groups = [torch.zeros((4, 4096), dtype=torch.int32) for _ in range(2)]
    groups[0][0, :3] = torch.tensor([2, 3, 4])
    groups[1][0, :3] = torch.tensor([7, 8, 9])

    def refresh(values):
        tables = adapter._tables(None, values)
        assert tables[0] is tables[2] and tables[1] is tables[3]
        gen._refresh_table(None, tables)

    refresh(groups * 2)
    assert gen.counters["page_table_refreshes"] == 2
    refresh([value.clone() for value in groups * 2])
    assert gen.counters["page_table_refreshes"] == 2
    groups[1][0, 3] = 10
    refresh(groups * 2)
    assert gen.counters["page_table_refreshes"] == 3
    assert gen.layer_page_tables[1][0, 3] == 10
    assert gen.layer_page_tables[0][0, 3] == 0
    assert gen._layer_page_hosts[0] is gen._layer_page_hosts[2]
    assert gen._layer_page_hosts[1] is gen._layer_page_hosts[3]

    # A remapped incoming row only replaces its scheduler slot; all other
    # slots retain the existing mappings. Distinct groups stay independent.
    incoming = [value[0:1].clone() for value in groups]
    tables = adapter._tables(None, incoming * 2, slots=[3])
    gen._refresh_table(None, tables)
    assert gen.counters["page_table_refreshes"] == 5
    assert gen.layer_page_tables[0][3, :3].tolist() == [2, 3, 4]
    assert gen.layer_page_tables[1][3, :3].tolist() == [7, 8, 9]
    assert gen.layer_page_tables[0][0, :3].tolist() == [2, 3, 4]
    inconsistent = [value.clone() for value in groups * 2]
    inconsistent[2][0, 0] = 15
    for call in (lambda: adapter._tables(None, inconsistent), lambda: gen._refresh_table(None, inconsistent)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("Inconsistent group mapping accepted")
    try:
        adapter.configure_kv_cache_groups([0] * 4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Live trace bindings were allowed to change")
    args.output.write_text(
        json.dumps(
            dict(
                status="pass",
                checks=[
                    "native grouping before allocation",
                    "one upload per changed group",
                    "zero uploads for unchanged tables",
                    "page growth",
                    "group isolation",
                    "slot remap preserves other rows",
                    "reject inconsistent mapping",
                    "reject reconfiguration after allocation",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
