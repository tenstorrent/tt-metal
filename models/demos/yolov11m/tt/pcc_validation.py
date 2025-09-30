"""
PCC (Pearson Correlation Coefficient) Validation Utilities for YOLOv11m

This module provides comprehensive PCC validation between PyTorch and TTNN implementations
to ensure model accuracy at each step of the pipeline.
"""

import torch
import ttnn
import numpy as np
from loguru import logger
from typing import Union, Optional, Tuple


class PCCValidator:
    """Comprehensive PCC validation for YOLOv11m model steps."""
    
    def __init__(self, min_pcc_threshold: float = 0.95):
        """
        Initialize PCC validator.
        
        Args:
            min_pcc_threshold: Minimum PCC value to consider as "good" correlation
        """
        self.min_pcc_threshold = min_pcc_threshold
        self.step_pccs = {}  # Track PCC for each step
        self.failed_steps = []  # Track steps that failed PCC threshold
        
    def calculate_pcc(self, 
                     pytorch_tensor: torch.Tensor, 
                     ttnn_tensor: Union[ttnn.Tensor, torch.Tensor], 
                     step_name: str = "unknown") -> float:
        """
        Calculate PCC between PyTorch and TTNN tensors.
        
        Args:
            pytorch_tensor: Reference PyTorch tensor
            ttnn_tensor: TTNN tensor to compare (will be converted to torch)
            step_name: Name of the step for logging
            
        Returns:
            float: PCC value (0.0 to 1.0)
        """
        # Convert TTNN tensor to PyTorch if needed
        if isinstance(ttnn_tensor, ttnn.Tensor):
            ttnn_torch = ttnn.to_torch(ttnn_tensor)
        else:
            ttnn_torch = ttnn_tensor
            
        # Ensure both tensors are on CPU and same shape
        pytorch_tensor = pytorch_tensor.detach().cpu()
        ttnn_torch = ttnn_torch.detach().cpu()
        
        # Handle shape mismatches
        if pytorch_tensor.shape != ttnn_torch.shape:
            logger.warning(f"[{step_name}] Shape mismatch: PyTorch {pytorch_tensor.shape} vs TTNN {ttnn_torch.shape}")
            # Try to reshape if total elements match
            if pytorch_tensor.numel() == ttnn_torch.numel():
                ttnn_torch = ttnn_torch.reshape(pytorch_tensor.shape)
            else:
                logger.error(f"[{step_name}] Cannot reshape - different total elements")
                return 0.0
        
        # Convert to same dtype
        if pytorch_tensor.dtype != ttnn_torch.dtype:
            ttnn_torch = ttnn_torch.to(pytorch_tensor.dtype)
        
        # Handle special cases
        if torch.all(torch.isnan(pytorch_tensor)) and torch.all(torch.isnan(ttnn_torch)):
            logger.warning(f"[{step_name}] Both tensors are NaN")
            return 1.0
            
        if torch.all(torch.isnan(pytorch_tensor)) or torch.all(torch.isnan(ttnn_torch)):
            logger.error(f"[{step_name}] One tensor is all NaN")
            return 0.0
        
        # Test if either is completely zero
        if torch.any(pytorch_tensor.bool()) != torch.any(ttnn_torch.bool()):
            logger.error(f"[{step_name}] One tensor is all zero")
            return 0.0
        
        # Handle exact equality
        if torch.equal(pytorch_tensor, ttnn_torch):
            return 1.0
        
        # Mask invalid values (nan/inf)
        def mask_invalid(x):
            x = x.clone()
            mask = torch.logical_or(
                torch.isnan(x), 
                torch.logical_or(torch.isinf(x), torch.isneginf(x))
            )
            x[mask] = 0
            return x
        
        pytorch_clean = mask_invalid(pytorch_tensor)
        ttnn_clean = mask_invalid(ttnn_torch)
        
        # Convert to float32 for correlation calculation
        if pytorch_clean.dtype == torch.bfloat16:
            pytorch_clean = pytorch_clean.to(torch.float32)
            ttnn_clean = ttnn_clean.to(torch.float32)
        
        # Calculate PCC using numpy
        try:
            pcc_matrix = np.ma.corrcoef(
                np.ma.masked_invalid(pytorch_clean.flatten().numpy()),
                np.ma.masked_invalid(ttnn_clean.flatten().numpy())
            )
            
            if isinstance(pcc_matrix, np.ma.core.MaskedConstant):
                return 1.0
                
            # Extract correlation coefficient (off-diagonal element)
            pcc_value = pcc_matrix[0, 1]
            
            if isinstance(pcc_value, np.ma.core.MaskedConstant):
                return 1.0
                
            return float(pcc_value)
            
        except Exception as e:
            logger.error(f"[{step_name}] PCC calculation failed: {e}")
            return 0.0
    
    def validate_step(self, 
                     pytorch_tensor: torch.Tensor, 
                     ttnn_tensor: Union[ttnn.Tensor, torch.Tensor], 
                     step_name: str,
                     show_stats: bool = True) -> Tuple[bool, float]:
        """
        Validate a single step and log results.
        
        Args:
            pytorch_tensor: Reference PyTorch tensor
            ttnn_tensor: TTNN tensor to compare
            step_name: Name of the step
            show_stats: Whether to show detailed statistics
            
        Returns:
            Tuple[bool, float]: (passed_threshold, pcc_value)
        """
        pcc_value = self.calculate_pcc(pytorch_tensor, ttnn_tensor, step_name)
        passed = pcc_value >= self.min_pcc_threshold
        
        # Store results
        self.step_pccs[step_name] = pcc_value
        if not passed:
            self.failed_steps.append(step_name)
        
        # Determine status emoji and color
        if pcc_value >= 0.99:
            status = "🟢 EXCELLENT"
        elif pcc_value >= 0.95:
            status = "🟡 GOOD"
        elif pcc_value >= 0.90:
            status = "🟠 ACCEPTABLE"
        else:
            status = "🔴 POOR"
        
        # Log results
        print(f"📊 [PCC] {step_name}: {pcc_value:.6f} {status}")
        
        if show_stats:
            # Convert tensors for stats
            if isinstance(ttnn_tensor, ttnn.Tensor):
                ttnn_torch = ttnn.to_torch(ttnn_tensor).detach().cpu()
            else:
                ttnn_torch = ttnn_tensor.detach().cpu()
                
            pytorch_cpu = pytorch_tensor.detach().cpu()
            
            print(f"    📈 PyTorch: min={pytorch_cpu.min():.6f}, max={pytorch_cpu.max():.6f}, mean={pytorch_cpu.mean():.6f}")
            print(f"    📈 TTNN:    min={ttnn_torch.min():.6f}, max={ttnn_torch.max():.6f}, mean={ttnn_torch.mean():.6f}")
            print(f"    📏 Shapes: PyTorch {pytorch_cpu.shape}, TTNN {ttnn_torch.shape}")
        
        return passed, pcc_value
    
    def get_summary(self) -> str:
        """Get a summary of all validation results."""
        if not self.step_pccs:
            return "📊 [PCC SUMMARY] No steps validated yet."
        
        total_steps = len(self.step_pccs)
        passed_steps = total_steps - len(self.failed_steps)
        avg_pcc = np.mean(list(self.step_pccs.values()))
        min_pcc = min(self.step_pccs.values())
        max_pcc = max(self.step_pccs.values())
        
        summary = [
            f"📊 [PCC SUMMARY] {passed_steps}/{total_steps} steps passed (≥{self.min_pcc_threshold})",
            f"    📈 Average PCC: {avg_pcc:.6f}",
            f"    📈 Min PCC: {min_pcc:.6f} | Max PCC: {max_pcc:.6f}",
        ]
        
        if self.failed_steps:
            summary.append(f"    🔴 Failed steps: {', '.join(self.failed_steps)}")
        else:
            summary.append(f"    🟢 All steps passed!")
        
        return "\n".join(summary)
    
    def reset(self):
        """Reset validation history."""
        self.step_pccs.clear()
        self.failed_steps.clear()


# Global validator instance
pcc_validator = PCCValidator(min_pcc_threshold=0.95)


def validate_pcc(pytorch_tensor: torch.Tensor, 
                ttnn_tensor: Union[ttnn.Tensor, torch.Tensor], 
                step_name: str,
                show_stats: bool = True,
                min_threshold: float = 0.95) -> Tuple[bool, float]:
    """
    Convenience function for PCC validation.
    
    Args:
        pytorch_tensor: Reference PyTorch tensor
        ttnn_tensor: TTNN tensor to compare
        step_name: Name of the step
        show_stats: Whether to show detailed statistics
        min_threshold: Minimum PCC threshold
        
    Returns:
        Tuple[bool, float]: (passed_threshold, pcc_value)
    """
    global pcc_validator
    pcc_validator.min_pcc_threshold = min_threshold
    return pcc_validator.validate_step(pytorch_tensor, ttnn_tensor, step_name, show_stats)


def get_pcc_summary() -> str:
    """Get summary of all PCC validations."""
    return pcc_validator.get_summary()


def reset_pcc_validation():
    """Reset PCC validation history."""
    pcc_validator.reset()
