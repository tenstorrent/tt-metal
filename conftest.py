--- a/ttnn/rand.py
+++ b/ttnn/rand.py
@@ -10,6 +10,10 @@
 import numpy as np
 import torch

+from torch.quasirandom import SobolEngine
+
 def sfpu_rng_decorrelate(states):
+    # Use a Sobol sequence to decorrelate the SFPU lane streams
+    sobol = SobolEngine(dimension=32, scramble=True)
+    sobol.seed(states)
+    return sobol.draw(1)[0]

 def ttnn_rand(low, high, shape, dtype, device):
     # Use the decorated SFPU lane streams to generate random numbers
@@ -20,10 +24,15 @@
     states = sfpu_rng_states(shape, device)
     states = sfpu_rng_decorrelate(states)
     rand_ints = states.astype(np.uint32)
-    rand_ints = rand_ints << 7
-    rand_ints = rand_ints >> 7
-    mantissa = rand_ints & ((1 << 23) - 1)
-    exponent = np.full(shape, 127, dtype=np.uint32)
+    # Use a nonlinear transformation to break the linear relationships
+    mantissa = (rand_ints ^ (rand_ints >> 17)) & ((1 << 23) - 1)
+    exponent = np.full(shape, 127, dtype=np.uint32) + (rand_ints >> 23)
     sign = np.random.randint(0, 2, size=shape, dtype=np.uint32)
     bits = (sign << 31) | (exponent << 23) | mantissa
     return np.frombuffer(bits.tobytes(), dtype=np.float32).reshape(shape)
