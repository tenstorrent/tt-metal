# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11m.tt.common import TtnnConv, Yolov11Conv2D, deallocate_tensors
from models.demos.yolov11m.tt.ttnn_yolov11_dwconv import TtnnDWConv


class TtnnOBB:
    """
    TTNN implementation of Oriented Bounding Box (OBB) detection head for YOLOv11.
    
    This module handles OBB detection with:
    - cv2: Box coordinate regression (same as standard Detect)
    - cv3: Class predictions using DWConv structure
    - cv4: Angle predictions for oriented bounding boxes
    - dfl: Distribution Focal Loss for coordinate regression
    """
    
    def __init__(self, device, parameter, conv_pt):
        """
        Initialize TtnnOBB module.
        
        Args:
            device: TTNN device
            parameter: Parameter configuration for OBB
            conv_pt: PyTorch OBB weights/parameters
        """
        self.device = device
        
        # cv2: Box coordinate regression (3 scales)
        # Scale 0 (256 -> 64 -> 64 -> 64)
        self.cv2_0_0 = TtnnConv(device, parameter.cv2[0][0], conv_pt.cv2[0][0], is_detect=True)
        self.cv2_0_1 = TtnnConv(device, parameter.cv2[0][1], conv_pt.cv2[0][1], is_detect=True)
        self.cv2_0_2 = Yolov11Conv2D(parameter.cv2[0][2], conv_pt.cv2[0][2], device=device, is_detect=True)
        print(f"🔍 cv2_0_2 weights range: {ttnn.to_torch(self.cv2_0_2.weight).min():.6f} to {ttnn.to_torch(self.cv2_0_2.weight).max():.6f}")
        if self.cv2_0_2.bias is not None:
            print(f"🔍 cv2_0_2 bias range: {ttnn.to_torch(self.cv2_0_2.bias).min():.6f} to {ttnn.to_torch(self.cv2_0_2.bias).max():.6f}")
        
        # Scale 1 (512 -> 64 -> 64 -> 64)
        self.cv2_1_0 = TtnnConv(device, parameter.cv2[1][0], conv_pt.cv2[1][0], is_detect=True)
        self.cv2_1_1 = TtnnConv(device, parameter.cv2[1][1], conv_pt.cv2[1][1], is_detect=True)
        self.cv2_1_2 = Yolov11Conv2D(parameter.cv2[1][2], conv_pt.cv2[1][2], device=device, is_detect=True)
        
        # Scale 2 (512 -> 64 -> 64 -> 64)
        self.cv2_2_0 = TtnnConv(device, parameter.cv2[2][0], conv_pt.cv2[2][0], is_detect=True)
        self.cv2_2_1 = TtnnConv(device, parameter.cv2[2][1], conv_pt.cv2[2][1], is_detect=True)
        self.cv2_2_2 = Yolov11Conv2D(parameter.cv2[2][2], conv_pt.cv2[2][2], device=device, is_detect=True)
        
        # cv3: Class predictions using DWConv structure (3 scales)
        # Scale 0: DWConv(256->256) -> Conv(256->256) -> DWConv(256->256) -> Conv(256->256) -> Conv2d(256->15)
        import pdb; pdb.set_trace()
        self.cv3_0_0_dw = TtnnDWConv(device, parameter.cv3[0][0][0], conv_pt.cv3[0][0][0], enable_act=True, is_detect=True)
        self.cv3_0_0_conv = TtnnConv(device, parameter.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True)
        self.cv3_0_1_dw = TtnnDWConv(device, parameter.cv3[0][1][0], conv_pt.cv3[0][1][0], enable_act=True, is_detect=True)
        self.cv3_0_1_conv = TtnnConv(device, parameter.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True)
        self.cv3_0_2 = Yolov11Conv2D(parameter.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True)
        print(f"🔍 cv3_0_2 weights range: {ttnn.to_torch(self.cv3_0_2.weight).min():.6f} to {ttnn.to_torch(self.cv3_0_2.weight).max():.6f}")
        if self.cv3_0_2.bias is not None:
            print(f"🔍 cv3_0_2 bias range: {ttnn.to_torch(self.cv3_0_2.bias).min():.6f} to {ttnn.to_torch(self.cv3_0_2.bias).max():.6f}")
        
        # Scale 1: DWConv(512->512) -> Conv(512->256) -> DWConv(256->256) -> Conv(256->256) -> Conv2d(256->15)
        self.cv3_1_0_dw = TtnnDWConv(device, parameter.cv3[1][0][0], conv_pt.cv3[1][0][0], enable_act=True, is_detect=True)
        self.cv3_1_0_conv = TtnnConv(device, parameter.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True)
        self.cv3_1_1_dw = TtnnDWConv(device, parameter.cv3[1][1][0], conv_pt.cv3[1][1][0], enable_act=True, is_detect=True)
        self.cv3_1_1_conv = TtnnConv(device, parameter.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True)
        self.cv3_1_2 = Yolov11Conv2D(parameter.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True)
        
        # Scale 2: DWConv(512->512) -> Conv(512->256) -> DWConv(256->256) -> Conv(256->256) -> Conv2d(256->15)
        self.cv3_2_0_dw = TtnnDWConv(device, parameter.cv3[2][0][0], conv_pt.cv3[2][0][0], enable_act=True, is_detect=True)
        self.cv3_2_0_conv = TtnnConv(device, parameter.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True)
        self.cv3_2_1_dw = TtnnDWConv(device, parameter.cv3[2][1][0], conv_pt.cv3[2][1][0], enable_act=True, is_detect=True)
        self.cv3_2_1_conv = TtnnConv(device, parameter.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True)
        self.cv3_2_2 = Yolov11Conv2D(parameter.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True)
        
        # cv4: Angle predictions (3 scales)
        # Scale 0: Conv(256->64) -> Conv(64->64) -> Conv2d(64->1)
        self.cv4_0_0 = TtnnConv(device, parameter.cv4[0][0], conv_pt.cv4[0][0], is_detect=True)
        self.cv4_0_1 = TtnnConv(device, parameter.cv4[0][1], conv_pt.cv4[0][1], is_detect=True)
        self.cv4_0_2 = Yolov11Conv2D(parameter.cv4[0][2], conv_pt.cv4[0][2], device=device, is_detect=True)
        
        # Scale 1: Conv(512->64) -> Conv(64->64) -> Conv2d(64->1)
        self.cv4_1_0 = TtnnConv(device, parameter.cv4[1][0], conv_pt.cv4[1][0], is_detect=True)
        self.cv4_1_1 = TtnnConv(device, parameter.cv4[1][1], conv_pt.cv4[1][1], is_detect=True)
        self.cv4_1_2 = Yolov11Conv2D(parameter.cv4[1][2], conv_pt.cv4[1][2], device=device, is_detect=True)
        
        # Scale 2: Conv(512->64) -> Conv(64->64) -> Conv2d(64->1)
        self.cv4_2_0 = TtnnConv(device, parameter.cv4[2][0], conv_pt.cv4[2][0], is_detect=True)
        self.cv4_2_1 = TtnnConv(device, parameter.cv4[2][1], conv_pt.cv4[2][1], is_detect=True)
        self.cv4_2_2 = Yolov11Conv2D(parameter.cv4[2][2], conv_pt.cv4[2][2], device=device, is_detect=True)
        
        # DFL: Distribution Focal Loss for coordinate regression
        self.dfl = Yolov11Conv2D(parameter.dfl.conv, conv_pt.dfl.conv, device=device, is_dfl=True)
        
        # Anchors and strides for coordinate transformation
        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, device, y1, y2, y3):
        """
        Forward pass through OBB detection head.
        
        Args:
            device: TTNN device
            y1: Feature map from scale 1 (smallest, highest resolution)
            y2: Feature map from scale 2 (medium resolution)  
            y3: Feature map from scale 3 (largest, lowest resolution)
            
        Returns:
            OBB detection output: [batch, 20, 8400] where 20 = 4(box) + 15(classes) + 1(angle)
        """
        
        # cv2: Box coordinate regression for all scales
        x1 = self.cv2_0_0(device, y1)
        x1 = self.cv2_0_1(device, x1)
        x1 = self.cv2_0_2(x1)
        
        x2 = self.cv2_1_0(device, y2)
        x2 = self.cv2_1_1(device, x2)
        x2 = self.cv2_1_2(x2)
        
        x3 = self.cv2_2_0(device, y3)
        x3 = self.cv2_2_1(device, x3)
        x3 = self.cv2_2_2(x3)
        
        # cv3: Class predictions with DWConv structure for all scales
        print(f"🔍 Debug cv3_0 - y1 input: {ttnn.to_torch(y1).min():.6f} to {ttnn.to_torch(y1).max():.6f}")
        
        x4 = self.cv3_0_0_dw(device, y1)
        print(f"🔍 Debug cv3_0 - after dw0: {ttnn.to_torch(x4).min():.6f} to {ttnn.to_torch(x4).max():.6f}")
        
        x4 = self.cv3_0_0_conv(device, x4)
        print(f"🔍 Debug cv3_0 - after conv0: {ttnn.to_torch(x4).min():.6f} to {ttnn.to_torch(x4).max():.6f}")
        
        x4 = self.cv3_0_1_dw(device, x4)
        print(f"🔍 Debug cv3_0 - after dw1: {ttnn.to_torch(x4).min():.6f} to {ttnn.to_torch(x4).max():.6f}")
        
        x4 = self.cv3_0_1_conv(device, x4)
        print(f"🔍 Debug cv3_0 - after conv1: {ttnn.to_torch(x4).min():.6f} to {ttnn.to_torch(x4).max():.6f}")
        
        x4 = self.cv3_0_2(x4)
        print(f"🔍 Debug cv3_0 - final output: {ttnn.to_torch(x4).min():.6f} to {ttnn.to_torch(x4).max():.6f}")
        
        x5 = self.cv3_1_0_dw(device, y2)
        x5 = self.cv3_1_0_conv(device, x5)
        x5 = self.cv3_1_1_dw(device, x5)
        x5 = self.cv3_1_1_conv(device, x5)
        x5 = self.cv3_1_2(x5)
        
        x6 = self.cv3_2_0_dw(device, y3)
        x6 = self.cv3_2_0_conv(device, x6)
        x6 = self.cv3_2_1_dw(device, x6)
        x6 = self.cv3_2_1_conv(device, x6)
        x6 = self.cv3_2_2(x6)
        
        # cv4: Angle predictions for all scales
        x7 = self.cv4_0_0(device, y1)
        x7 = self.cv4_0_1(device, x7)
        x7 = self.cv4_0_2(x7)
        
        x8 = self.cv4_1_0(device, y2)
        x8 = self.cv4_1_1(device, x8)
        x8 = self.cv4_1_2(x8)
        
        x9 = self.cv4_2_0(device, y3)
        x9 = self.cv4_2_1(device, x9)
        x9 = self.cv4_2_2(x9)
        
        # Convert tensors to interleaved memory layout for concatenation
        x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)
        x7 = ttnn.sharded_to_interleaved(x7, memory_config=ttnn.L1_MEMORY_CONFIG)
        x8 = ttnn.sharded_to_interleaved(x8, memory_config=ttnn.L1_MEMORY_CONFIG)
        x9 = ttnn.sharded_to_interleaved(x9, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # Concatenate box coordinates and class predictions for each scale
        y1_box_cls = ttnn.concat((x1, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2_box_cls = ttnn.concat((x2, x5), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3_box_cls = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # Concatenate all scales for box coordinates and class predictions
        y_box_cls = ttnn.concat((y1_box_cls, y2_box_cls, y3_box_cls), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y_box_cls = ttnn.squeeze(y_box_cls, dim=0)
        
        # Split into box coordinates (64 channels) and class predictions (15 channels)
        ya, yb = y_box_cls[:, :, :64], y_box_cls[:, :, 64:79]  # 64 + 15 = 79
        
        # Clean up intermediate tensors
        deallocate_tensors(y1_box_cls, y2_box_cls, y3_box_cls, x1, x2, x3, x4, x5, x6, y_box_cls)
        
        # Process box coordinates through DFL
        ya = ttnn.reallocate(ya)
        yb = ttnn.reallocate(yb)
        ya = ttnn.reshape(ya, (ya.shape[0], ya.shape[1], 4, 16))
        ya = ttnn.softmax(ya, dim=-1)
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        c = self.dfl(ya)
        ttnn.deallocate(ya)
        
        # Transform DFL output to coordinates
        c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]
        
        # Apply anchor and stride transformations
        anchor, strides = self.anchors, self.strides
        anchor = ttnn.to_memory_config(anchor, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)
        c1 = anchor - c1
        c2 = anchor + c2
        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)
        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, strides)
        
        # Process class predictions - transpose to [batch, 15, N] for concat
        yb = ttnn.permute(yb, (0, 2, 1))  # [batch, H*W, 15] -> [batch, 15, H*W]
        yb = ttnn.sigmoid(yb)
        
        # Process angle predictions - reshape and concat to get [batch, 1, N]
        x7 = ttnn.reshape(x7, (x7.shape[0], x7.shape[1], x7.shape[2] * x7.shape[3]))
        x8 = ttnn.reshape(x8, (x8.shape[0], x8.shape[1], x8.shape[2] * x8.shape[3]))
        x9 = ttnn.reshape(x9, (x9.shape[0], x9.shape[1], x9.shape[2] * x9.shape[3]))
        angle_pred = ttnn.concat((x7, x8, x9), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # Clean up intermediate tensors
        deallocate_tensors(c, z1, z2, c1, c2, anchor, strides, x7, x8, x9)
        
        # Reallocate for final concatenation
        z = ttnn.reallocate(z)
        yb = ttnn.reallocate(yb)
        angle_pred = ttnn.reallocate(angle_pred)
        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        angle_pred = ttnn.to_layout(angle_pred, layout=ttnn.ROW_MAJOR_LAYOUT)
        
        
        # Final output: [box_coords(4), class_preds(15), angle_pred(1)] = 20 channels total
        out = ttnn.concat((z, yb, angle_pred), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        
        # Clean up final intermediate tensors
        deallocate_tensors(yb, z, angle_pred)
        
        return out
