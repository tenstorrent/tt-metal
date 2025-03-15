# Vanilla UNet in TT-NN

## Contents
- [Vanilla UNet in TT-NN](#)
  - [Contents](#contents)
  - [1.Overview](#1-overview)
  - [2.Vanilla UNet TT-NN Optimization Techniques](#2-vanilla-unet-tt-nn-optimization-techniques)
    - [2.1 Sharding on all relevant OPs](#21-sharding-on-all-relevant-ops)

## 1. Overview

## 2. Vanilla UNet TT-NN Optimization Techniques

The implemented optimization techniques in TT-NN compared to the conventional flow are:
### 2.1 Sharding on all relevant OPs
  - Applying sharding techniques to harvest the optimum utilization of the computation OPs, by eliminating the need for data movement inter-tensix-cores between the consecutive OPs.
  - For more details, please refer to the [related tech-report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/tensor_layouts/tensor_layouts.md#42-sharding)
  - Sharding Concepts
![Sharding Concept](images/sharding_concept.png)
  - Illustrative example
![Sharding Example](images/sharding_example.png)
