#!/bin/bash

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"

mpirun hostname
