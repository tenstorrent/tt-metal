# SPDX-License-Identifier: Apache-2.0
"""Whole-role microbenchmark windows plus DRAM/matmul diagnostics."""
import argparse,csv,json,re,subprocess,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('results',type=Path);args=p.parse_args()
source=sorted(args.root.glob('reports/*/ops_perf_results*.csv'))[-1]
rows=list(csv.DictReader(source.open()))
header=(args.root/'.logs/profile_log_device.csv').open().readline();mhz=float(re.search(r'CHIP_FREQ\[MHz\]: (\d+)',header)[1])
results=json.loads(args.results.read_text())
for result in results:
 name=f"MICRO_{result['role']}_R{result['readers']}_{result['repeat']}"
 start=next(i for i,r in enumerate(rows) if r['OP CODE']==name)
 end=next(i for i,r in enumerate(rows) if i>start and r['OP CODE']==name+'_END')
 ops=[r for r in rows[start+1:end] if r['OP TYPE']=='tt_dnn_device'];mm=max((r for r in ops if r['OP CODE']=='MatmulDeviceOperation'),key=lambda r:float(r['DEVICE KERNEL DURATION [ns]']))
 per_device={d:(max(int(r['DEVICE FW END CYCLE']) for r in ops if r['DEVICE ID']==d)-min(int(r['DEVICE FW START CYCLE']) for r in ops if r['DEVICE ID']==d))/mhz for d in set(r['DEVICE ID'] for r in ops)}
 us=max(per_device.values())
 result['per_device_envelope_us']=per_device
 k,n=result['k'],result['n'];physical=k*((n+255)//256)*256/1024*576;logical=k*result['logical_n']/1024*576
 # BFLOAT4_B stores a 1024-element tile in 576 bytes, including exponent headers.
 result.update(device_us=us,matmul_kernel_us=float(mm['DEVICE KERNEL DURATION [ns]'])/1000,
  physical_bytes=physical,logical_bytes=logical,physical_gbs=physical/us/1000,logical_gbs=logical/us/1000,
  dram_peak_pct=100*(physical/us/1000)/512,physical_row_bytes_per_reader=((n+255)//256)*576/result['readers'],
  noc_max_burst_bytes=16384,estimated_bursts_per_reader_row=math.ceil((((n+255)//256)*576/result["readers"])/16384),
  input_storage_cores=result['config'].get('cores',8),attributes=mm['ATTRIBUTES'],
  risc_us={r:float(mm[f'DEVICE {r} KERNEL DURATION [ns]'])/1000 for r in ('BRISC','NCRISC','TRISC0','TRISC1','TRISC2')},
  source=str(source))
args.results.with_suffix('.device.json').write_text(json.dumps(results,indent=2)+'\n')
for raw in args.root.rglob('*.csv'):
 if raw.stat().st_size>64*1024**2:subprocess.run(['zstd','-q','-T2','--rm',str(raw)],check=True)
