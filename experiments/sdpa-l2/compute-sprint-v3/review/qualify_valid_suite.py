"""Bounded batches for the frozen B validity candidate; parent supplies one device lock."""

import argparse
from types import SimpleNamespace

import qualify_valid as Q


SUITES = {
    "short": [
        ("k1",512,512,1,20260918,"normal,constant_v,zero_v"),
        ("k2",1024,1024,2,20260918,"normal,uniform,constant_v_scaled,common_v,identity_transitions"),
        ("k8",1024,4096,2,20260918,"normal,growing_max,scaled_qk,outliers,common_q,common_k,common_qk,common_v,identity_transitions,uniform_constant_v_scaled"),
    ],
    "wide": [
        ("32k",256,32768,1,20260918,"normal,uniform,constant_v,constant_v_scaled,common_v,common_q,common_k,common_qk,outliers,scaled_qk,growing_max,zero_v"),
        ("256k",256,262144,1,20260918,"normal,uniform,constant_v,common_v,repeated_kv,growing_max"),
    ],
    "heldout": [
        ("heldout8k",2048,8192,2,20260926,"normal,uniform,constant_v_scaled,uniform_constant_v_scaled,common_v,common_q,common_k,common_qk,outliers,scaled_qk,growing_max,identity_transitions,zero_v"),
        ("heldout32k",512,32768,1,20260927,"normal,growing_max,common_v,scaled_qk,outliers,common_qk"),
    ],
}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--suite",choices=tuple(SUITES),required=True)
    p.add_argument("--label-prefix",required=True)
    args=p.parse_args()
    for name,q,k,cores,seed,distributions in SUITES[args.suite]:
        Q.run(SimpleNamespace(label=f"{args.label_prefix}-{name}",q_length=q,k_length=k,
                              cores=cores,seed=seed,distributions=distributions))
