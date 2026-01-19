class ReturnInInitE0101:
    def __init__(self, value):
        # Should trigger "return-in-init"
        return value
