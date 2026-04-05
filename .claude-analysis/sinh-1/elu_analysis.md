# SFPU Analysis: elu

## Overview
ELU(x) = x if x >= 0, alpha*(exp(x)-1) if x < 0

## SFPU Kernel
Uses exp-based computation with conditional branching. Shows exp(x) - 1 pattern.

## Relevance to sinh
Shows exp-subtraction pattern useful for understanding (exp(x) - exp(-x)).
