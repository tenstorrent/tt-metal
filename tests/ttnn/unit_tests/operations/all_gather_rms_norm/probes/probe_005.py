import ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache

c = _FabricMeshCache()
m = c.get((2, 2))
print("OPEN1 OK", m.shape)
c.close()
m = c.get((2, 2))
print("OPEN2 OK", m.shape)
c.close()
print("DONE")
