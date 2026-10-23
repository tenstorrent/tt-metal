#!/bin/bash
BARRIER_DIR="/nfs/$USER/.barrier"
rm -rf $BARRIER_DIR
mkdir -p $BARRIER_DIR

# Execute with output capture
parallel-ssh -i -H "sjc1-tt-qb-01 sjc1-tt-qb-02 sjc1-tt-qb-03 sjc1-tt-qb-04" \
	"cd /nfs/$USER/tt-smi && source .venv/bin/activate && \
   touch $BARRIER_DIR/\$(hostname) && \
   while [ \$(ls $BARRIER_DIR | wc -l) -lt 4 ]; do sleep 0.01; done && \
   tt-smi -r"

# Cleanup
rm -rf $BARRIER_DIR
