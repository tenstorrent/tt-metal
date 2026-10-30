# SPDX-License-Identifier: Apache-2.0
"""Disambiguate matmul/reduction fidelity in multi-kernel profiler rows."""
import hashlib
import json
from pathlib import Path
import re


def fused_matmul_fidelity(row):
    assert 'compute_kernel_config=ComputeKernelConfig(math_fidelity=LoFi;' in row['ATTRIBUTES']
    repo = Path(__file__).resolve().parents[4]
    base = Path(__file__).resolve().parents[1] / 'doc/optimized_multichip_decoder'
    evidence = base / 'kernel_fidelity_evidence'
    evidence.mkdir(exist_ok=True)
    kernels = []
    for kernel in re.findall(r"'([^']+)'", row['COMPUTE KERNEL HASH']):
        name, digest = kernel.rstrip('/').split('/')
        saved = evidence / f'{name}_{digest}.h'
        matches = sorted((repo / '.cache/tt-metal-cache').glob(f'*/kernels/{name}/{digest}/chlkc_descriptors.h'))
        if matches:
            contents = matches[0].read_bytes()
            assert all(p.read_bytes() == contents for p in matches), kernel
            if saved.exists():
                assert saved.read_bytes() == contents, kernel
            else:
                saved.write_bytes(contents)
        assert saved.exists(), f'Missing compiled descriptor for captured kernel {kernel}'
        contents = saved.read_bytes()
        codes = set(re.findall(r'MATH_FIDELITY = static_cast<ckernel::MathFidelity>\((\d+)\)', contents.decode()))
        assert len(codes) == 1, kernel
        code = int(next(iter(codes)))
        if name.startswith('bmm_'):
            assert code == 0, (kernel, code)
        kernels.append(dict(kernel=kernel,math_fidelity_enum=code,
                            descriptor=str(saved.relative_to(base)),sha256=hashlib.sha256(contents).hexdigest(),
                            cache_source=str(matches[0]) if matches else None))
    assert any(k['kernel'].startswith('bmm_') for k in kernels)
    receipt = dict(matmul_fidelity='LoFi',scalar_profiler_fidelity=row['MATH FIDELITY'],kernels=kernels,
                   explanation='process_ops_logs.py overwrites scalar fidelity for each compute kernel; the reduction is last. Kernel descriptors and captured matmul attributes independently prove LoFi.')
    return receipt
