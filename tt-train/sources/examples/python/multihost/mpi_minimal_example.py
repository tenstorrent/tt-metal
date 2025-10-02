import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')

import ttml

ttml.autograd.AutoContext.get_instance().initialize_distributed_context(*sys.argv)

distributed_ctx = ttml.autograd.AutoContext.get_instance().get_distributed_context()
print("world size:", distributed_ctx.size(), "rank:", distributed_ctx.rank())
