"""Set inherited CPU affinity before starting any owner or native child."""

import os
import sys

allowed = os.sched_getaffinity(0)
assert allowed
os.sched_setaffinity(0, {min(allowed)})
assert len(os.sched_getaffinity(0)) == 1
os.execv(sys.argv[1], sys.argv[1:])
