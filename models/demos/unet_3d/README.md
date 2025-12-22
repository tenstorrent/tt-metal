make sure that reshape and layout does not cause unnecesary new alocaitons

rehspae: make surethe last dim stays the same


Use the combinat of 2d ops to implement 3d ops, e.g. maxpool and up sample

maxpool uses 2x the mory of input so memory is bottleneck for this op, if therei s inplace version thta
does not equire new allocation, would be make larger inputs possible to do in one pass.
