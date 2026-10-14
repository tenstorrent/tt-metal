# Op level: Q (the port) against P (the head without it), P2 repeats P (A/A): every changed unary op and the production
# chains at DRAM 1x1x1024x1024 and height-sharded, model shapes, relu as a cheap reference.
ORDERS=("P Q P2" "P2 Q P" "Q P P2" "P2 P Q")
CASES=""
