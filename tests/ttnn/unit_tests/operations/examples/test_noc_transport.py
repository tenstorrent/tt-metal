# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic DRAM transport characterization; operation-independent and exact-output checked."""

import os

for key, value in {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_LOGGER_LEVEL": "error",
}.items():
    os.environ.setdefault(key, value)
import csv, io, json, re, socket, statistics, hashlib
from pathlib import Path
from datetime import datetime, timezone
import pytest, torch, ttnn

SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
void kernel_main(){
 constexpr uint32_t bytes=get_compile_time_arg_val(0),pages=64,iters=4;
 constexpr uint32_t part=get_compile_time_arg_val(1),parts=get_compile_time_arg_val(2),stride=get_compile_time_arg_val(3),batch=get_compile_time_arg_val(4),write=get_compile_time_arg_val(5);
 constexpr auto ia=TensorAccessorArgs<6>();constexpr auto oa=TensorAccessorArgs<ia.next_compile_time_args_offset()>();
 auto src=TensorAccessor(ia,get_arg_val<uint32_t>(0),bytes);
 auto dst=TensorAccessor(oa,get_arg_val<uint32_t>(1),bytes);
 uint32_t core=get_arg_val<uint32_t>(2),scratch=get_write_ptr(part);
 // Initialize source L1 directly, so preparation generates no DRAM traffic.
 if constexpr(write){
  for(uint32_t b=0;b<pages;b+=batch)for(uint32_t j=0;j<batch/parts;++j){
   uint32_t page=(core*pages+b+part*(batch/parts)+j)*stride;
   auto ptr=reinterpret_cast<volatile uint32_t*>(scratch+(b/parts+j)*bytes);
   for(uint32_t w=0;w<bytes/4;++w)ptr[w]=(page*(bytes/4)+w)%65521+1;
  }
 }
 auto issue=[&](uint32_t b){
  for(uint32_t j=0;j<batch/parts;++j){
   uint32_t page=(core*pages+b+part*(batch/parts)+j)*stride;
   uint32_t ptr=scratch+(b/parts+j)*bytes;
   if constexpr(write){noc_async_write(ptr,dst.get_noc_addr(page),bytes);}
   else{noc_async_read(src.get_noc_addr(page),ptr,bytes);}
  }
 };
 {
 DeviceZoneScopedN("TRANSPORT");
 for(uint32_t it=0;it<iters;++it)for(uint32_t b=0;b<pages;b+=batch){
  if(it==0&&b==0){DeviceZoneScopedN("ISSUE");issue(b);}else{issue(b);}
  if constexpr(write){
   if(it==0&&b==0){
    {DeviceZoneScopedN("FLUSH");noc_async_writes_flushed();}
    {DeviceZoneScopedN("ACK");noc_async_write_barrier();}
   }else{noc_async_write_barrier();}
  }else{noc_async_read_barrier();}
 }
 }
 // Read verification writes occur outside timing. Write-only probes have no post-scope data traffic.
 if constexpr(!write){
  for(uint32_t b=0;b<pages;b+=batch)for(uint32_t j=0;j<batch/parts;++j){
   uint32_t page=(core*pages+b+part*(batch/parts)+j)*stride;
   noc_async_write(scratch+(b/parts+j)*bytes,dst.get_noc_addr(page),bytes);
  }
  noc_async_write_barrier();
 }
}
"""


def parse(device):
    ttnn.ReadDeviceProfiler(device)
    programs = [p for ps in ttnn.get_latest_programs_perf_data().values() for p in ps]
    assert len(programs) == 1
    uid = programs[0].program_execution_uid.runtime_id
    with open("generated/profiler/.logs/profile_log_device.csv") as f:
        mhz = float(re.search(r"CHIP_FREQ\[MHz\]: (\d+)", f.readline()).group(1))
        fields = next(csv.reader([f.readline()], skipinitialspace=True))
        start = f.tell()
        previous = getattr(parse, "offset", start)
        f.seek(0, 2)
        end = f.tell()
        f.seek(previous if start <= previous <= end else start)
        data = f.read()
        parse.offset = f.tell()
    zones = {}
    for r in csv.DictReader(io.StringIO(data), fieldnames=fields, skipinitialspace=True):
        if int(r["run host ID"]) != uid or r["zone name"] not in ["TRANSPORT", "ISSUE", "FLUSH", "ACK"]:
            continue
        key = (r["core_x"], r["core_y"], r["RISC processor type"])
        zones.setdefault(key, {}).setdefault(r["zone name"], {})[r["type"]] = int(r["time[cycles since reset]"])
    return mhz, zones


CASES = (
    json.loads(Path(os.environ["NOC_TRANSPORT_CASES"]).read_text())
    if "NOC_TRANSPORT_CASES" in os.environ
    else [dict(cores=1, payload_bytes=128, batch=16, mode="noc1", banks="balanced", traffic="write")]
)


@pytest.mark.parametrize("case", CASES)
def test_transport(device, case):
    cores, payload, batch = case["cores"], case["payload_bytes"], case["batch"]
    parts = 2 if case["mode"] == "split" else 1
    grid = device.compute_with_storage_grid_size()
    nb = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(cores)]
    stride = nb if case["banks"] == "hotspot" else 1
    pages = 64
    assert pages % batch == 0 and batch % parts == 0 and cores <= grid.x * grid.y
    values = (torch.arange(cores * pages * stride * (payload // 4), dtype=torch.int64) % 65521 + 1).reshape(
        -1, payload // 4
    )
    inp = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.from_torch(
        torch.zeros_like(values),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    ra = ttnn.RuntimeArgs()
    for i, c in enumerate(coords):
        ra[c.x][c.y] = [inp.buffer_address(), out.buffer_address(), i]
    kernels = []
    cbs = []
    for part in range(parts):
        noc = part if parts == 2 else int(case["mode"][-1])
        write = part == 1 if case["traffic"] == "mixed" else case["traffic"] == "write"
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=SOURCE,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                runtime_args=ra,
                compile_time_args=[payload, part, parts, stride, batch, int(write)]
                + list(ttnn.TensorAccessorArgs(inp).get_compile_time_args())
                + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
                config=ttnn.ReaderConfigDescriptor() if noc == 0 else ttnn.WriterConfigDescriptor(),
            )
        )
        cbs.append(
            ttnn.CBDescriptor(
                total_size=pages // parts * payload,
                core_ranges=ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=part, data_format=ttnn.uint32, page_size=payload)
                ],
            )
        )
    desc = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs)
    run = lambda: ttnn.generic_op([inp, out], desc)
    run()
    ttnn.ReadDeviceProfiler(device)
    samples = []
    issue = {}
    stages = {}
    raw = []
    trials = int(os.environ.get("NOC_TRANSPORT_TRIALS", "5"))
    for _ in range(trials):
        run()
        mhz, zones = parse(device)
        assert len(zones) == cores * parts
        durations = []
        for key, z in zones.items():
            span = lambda name: z[name]["ZONE_END"] - z[name]["ZONE_START"]
            durations.append(span("TRANSPORT") / 4)
            issue.setdefault(key[2], []).append(span("ISSUE") / (batch // parts))
            for name in ["FLUSH", "ACK"]:
                if name in z:
                    stages.setdefault(key[2] + "_" + name, []).append(span(name))
        samples.append(max(durations))
        raw.append({",".join(k): v for k, v in zones.items()})
    expected = torch.zeros_like(values)
    expected[::stride] = values[::stride]
    torch.testing.assert_close(ttnn.to_torch(out).to(torch.int64), expected, rtol=0, atol=0)
    result = dict(
        **case,
        pages=pages,
        iterations=4,
        trials=trials,
        clock_mhz=mhz,
        bank_count=nb,
        logical_coords=[[c.x, c.y] for c in coords],
        placement="row",
        host=socket.gethostname(),
        arch=str(device.arch()),
        utc=datetime.now(timezone.utc).isoformat(),
        samples_cycles=samples,
        median_cycles=statistics.median(samples),
        std_cycles=statistics.pstdev(samples),
        issue_cycles={k: statistics.median(v) for k, v in issue.items()},
        first_batch_stage_cycles={k: statistics.median(v) for k, v in stages.items()},
        raw_zones=raw,
        correct=True,
        correctness="exact_entire_output_including_unwritten_pages",
        metric="max_per_core_per_risc_scoped_transport_cycles_per_iteration",
        kernel_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        caveat=(
            "read verification writes outside the scope can overlap late readers/writers; write-only cases have none"
            if case["traffic"] != "write"
            else "per-core local timing; no cross-core start barrier"
        ),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "raw_zones"}))
    path = Path(os.environ.get("NOC_TRANSPORT_REPORT", "/tmp/noc-write-validation/measured.jsonl"))
    with path.open("a") as f:
        f.write(json.dumps(result) + "\n")
    ttnn.deallocate(inp)
    ttnn.deallocate(out)
