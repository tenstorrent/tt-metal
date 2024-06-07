import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        # left side of graph
        self.c1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # there will be concat here
        # there will be 6 CBRs on the left bellow concat
        self.c2 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.c3 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b3 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.c4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.c5 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.c6 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b6 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.c7 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b7 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # last stand-alone conv on left - generates output1
        self.c8 = nn.Conv2d(512, 255, 1, 1, 0, bias=False)

        # right side of graph
        self.c9 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b9 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # stand-alone conv on right - generates output2
        self.c10 = nn.Conv2d(256, 255, 1, 1, 0, bias=False)

        # CBR bellow output2 on the right graph BUT takes the output of self.relu(6)
        self.c11 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.b11 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        # there will be a concat here
        self.c12 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b12 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.c13 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b13 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

        self.c14 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b14 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.c15 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b15 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

        self.c16 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b16 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)

        self.c17 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b17 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

        # last stand alone conv from right side of graph. outputs output3
        self.c18 = nn.Conv2d(1024, 255, 1, 1, 0, bias=False)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_r = self.relu(x1_b)

        # this is a dummy input for now. the real input will come from the Neck
        outfromNeck1 = torch.ones(1, 256, 20, 20)
        cat1 = torch.cat([outfromNeck1, x1_r], dim=1)
        x2 = self.c2(cat1)
        x2_b = self.b2(x2)
        x2_r = self.relu(x2_b)

        x3 = self.c3(x2_r)
        x3_b = self.b3(x3)
        x3_r = self.relu(x3_b)

        x4 = self.c4(x3_r)
        x4_b = self.b4(x4)
        x4_r = self.relu(x4_b)

        x5 = self.c5(x4_r)
        x5_b = self.b5(x5)
        x5_r = self.relu(x5_b)

        x6 = self.c6(x5_r)
        x6_b = self.b6(x6)
        x6_r = self.relu(x6_b)

        x7 = self.c7(x6_r)
        x7_b = self.b7(x7)
        x7_r = self.relu(x7_b)

        # this generates output1
        x8 = self.c8(x7_r)

        x9 = self.c9(input)
        x9_b = self.b9(x9)
        x9_r = self.relu(x9_b)

        # this generates output2
        x10 = self.c10(x9_r)

        x11 = self.c11(x6_r)
        x11_b = self.b11(x11)
        x11_r = self.relu(x11_b)

        # this is a dummy input for now. the real input will come from the Neck
        outfromNeck2 = torch.ones(1, 512, 10, 10)
        cat2 = torch.cat([outfromNeck2, x11_r], dim=1)

        x12 = self.c12(cat2)
        x12_b = self.b12(x12)
        x12_r = self.relu(x12_b)

        x13 = self.c13(x12_r)
        x13_b = self.b13(x13)
        x13_r = self.relu(x13_b)

        x14 = self.c14(x13_r)
        x14_b = self.b14(x14)
        x14_r = self.relu(x14_b)

        x15 = self.c15(x14_r)
        x15_b = self.b15(x15)
        x15_r = self.relu(x15_b)
        num_channels = x15_r.size(1)  # Get the number of channels
        print("the number of channels in x15_r are: ", num_channels)
        print("self.c16: ", self.c16)
        x16 = self.c16(x15_r)
        x16_b = self.b16(x16)
        x16_r = self.relu(x16_b)

        # generates output3
        x17 = self.c17(x16_r)
        x17_b = self.b17(x17)
        x17_r = self.relu(x17_b)

        x18 = self.c18(x17_r)

        return x8, x10, x18
