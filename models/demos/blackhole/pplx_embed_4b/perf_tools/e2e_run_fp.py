# e2e with full_pipeline=True (forward + pooling + I/O inside one traced replay)
import sys

from models.demos.blackhole.pplx_embed_4b.demo._common import standalone_main

standalone_main(batch_size=int(sys.argv[1]), seq_len=512, iterations=int(sys.argv[2]), full_pipeline=True)
