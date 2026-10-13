# Op level: G (the head with the gate) against P (previous head), P2 repeats P (A/A); every changed unary op at DRAM
# 1x1x1024x1024 and height-sharded, model shapes, relu as a cheap reference.
ORDERS=("P G P2" "P2 G P" "G P P2" "P2 P G")
CASES=""
