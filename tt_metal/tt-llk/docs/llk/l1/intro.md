# Introduction to Tenstorrent's Low-Level Kernels (LLKs)

## Overview

Each Tenstorrent chip consists of a grid of Tensix Cores that work together to run advanced ML models.
Low-Level Kernels (LLKs) are essential software components of our AI ecosystem.
These kernels enable AI model operation on Tensix cores at high performance. They act as the first software layer above the hardware and provide direct access to its computational capabilities.
Developers use low-level kernels and primitives to efficiently build complex operations.

![Tensix Core to Chip](images/LLK-L1-Tensix-1-25_official.png)

## Purpose and Functionality

LLKs control the Tensix Engine, the fundamental computational unit of the Tensix Core, using the custom Tensix instruction set (Tensix ISA) extensions. LLKs also simplify usage of the Tensix ISA, allowing operations to be performed with maximum efficiency and fully utilizing the hardware's parallelism and optimized compute resources.

As the foundational software layer, LLKs provide a direct interface for higher-level applications to harness the full power of the Tensix core, encapsulating hardware specifics into software primitives. They enable low-level tasks like data movement and mathematical operations to be executed at peak performance.

## LLKs as Part of Tenstorrent's Software Stack

LLKs are a core part of Tenstorrent’s software ecosystem. In addition to enabling standalone software development, they are designed to work seamlessly with other tools such as:

- [**TT-Metalium™**](https://github.com/tenstorrent/tt-metal): An open-source SDK that facilitates the development of custom kernels, offering low-level access to Tenstorrent hardware and leveraging LLKs for optimized performance.
- [**TT-Forge™**](https://github.com/tenstorrent/tt-forge): A compiler that bridges machine learning frameworks with Tenstorrent hardware, using LLKs to optimize the execution of high-level models on Tensix cores.

Together, these tools help developers build and optimize machine learning models and other high-performance applications tailored to Tenstorrent hardware.

![LLK in Tenstorrent Stack](images/LLK-L1-Tensix-3-25_official.png)

## Development and Testing

The LLK repository provides the necessary resources for developing and testing kernels. It also includes a testing environment which validates that the LLKs function correctly on platforms like Wormhole and Blackhole. By targeting a single Tensix core, LLKs are optimized for high-performance execution under real-world workloads, and ensure that developers can fully leverage the hardware’s capabilities. The LLK repository has its own independent CI and test infrastructure, making it easier for developers to contribute.
