import ttnn
E=ttnn.experimental
print([x for x in dir(E) if 'Precision' in x])
c=ttnn.ComputeConfigDescriptor()
print([a for a in dir(c) if not a.startswith('_')])
print(c.dst_full_sync_en, c.math_approx_mode)
