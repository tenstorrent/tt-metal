"""
Lifting Wavelet Transform (LWT) Implementation

Pure Python implementation of LWT and Inverse LWT
with support for multiple wavelet families

版权声明：MIT License | Copyright (c) 2026 思捷娅科技 (SJYKJ)
"""

import numpy as np
from typing import Tuple, List, Union


class LiftingWaveletTransform:
    """
    Lifting Wavelet Transform implementation
    
    Supports:
    - Haar wavelet
    - Daubechies-4 (Db4) wavelet
    - Cohen-Daubechies-Feauveau (CDF) 9/7 wavelet
    """
    
    def __init__(self, wavelet: str = 'haar', levels: int = 1):
        """
        Initialize LWT
        
        Args:
            wavelet: Wavelet type ('haar', 'db4', 'cdf97')
            levels: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels
        self.filters = self._get_filters(wavelet)
    
    def _get_filters(self, wavelet: str) -> dict:
        """Get prediction and update filters for wavelet"""
        
        if wavelet == 'haar':
            return {
                'predict': [1.0],
                'update': [0.5],
                'norm': 1.0 / np.sqrt(2)
            }
        elif wavelet == 'db4':
            # Daubechies-4 coefficients
            return {
                'predict': [0.4829629131445341, 0.8365163037378079],
                'update': [-0.2241438680420134, 0.1294095225512604],
                'norm': 1.0
            }
        elif wavelet == 'cdf97':
            # CDF 9/7 wavelet (used in JPEG2000)
            return {
                'predict': [-1.586134342, -0.05298011854],
                'update': [0.8829110762, 0.4435068522],
                'norm': 1.0
            }
        else:
            raise ValueError(f"Unknown wavelet: {wavelet}")
    
    def _split(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split signal into even and odd samples"""
        even = signal[0::2]
        odd = signal[1::2]
        return even, odd
    
    def _predict(self, even: np.ndarray, odd: np.ndarray) -> np.ndarray:
        """Predict step: predict odd from even"""
        d = odd.copy()
        for i, coef in enumerate(self.filters['predict']):
            if i == 0:
                d[:-1] -= coef * even[1:]
            else:
                d[:-1] -= coef * even[:-1]
        return d
    
    def _update(self, even: np.ndarray, detail: np.ndarray) -> np.ndarray:
        """Update step: update even using detail"""
        s = even.copy()
        for i, coef in enumerate(self.filters['update']):
            if i == 0:
                s[1:] += coef * detail[:-1]
            else:
                s[1:] += coef * detail[1:]
        return s
    
    def _forward_level(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single level forward LWT"""
        even, odd = self._split(signal)
        detail = self._predict(even, odd)
        approx = self._update(even, detail)
        
        # Normalization
        norm = self.filters['norm']
        approx = approx * norm
        detail = detail / norm
        
        return approx, detail
    
    def _inverse_level(self, approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
        """Single level inverse LWT"""
        # Denormalization
        norm = self.filters['norm']
        approx = approx / norm
        detail = detail * norm
        
        # Inverse update
        even = approx.copy()
        for i, coef in enumerate(self.filters['update']):
            if i == 0:
                even[1:] -= coef * detail[:-1]
            else:
                even[1:] -= coef * detail[1:]
        
        # Inverse predict
        odd = detail.copy()
        for i, coef in enumerate(self.filters['predict']):
            if i == 0:
                odd[:-1] += coef * even[1:]
            else:
                odd[:-1] += coef * even[:-1]
        
        # Merge
        signal = np.zeros(len(even) + len(odd))
        signal[0::2] = even
        signal[1::2] = odd[:len(even)]
        
        return signal
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Forward LWT
        
        Args:
            signal: Input signal (1D or 2D)
            
        Returns:
            Wavelet coefficients
        """
        if signal.ndim == 1:
            return self._transform_1d(signal)
        elif signal.ndim == 2:
            return self._transform_2d(signal)
        else:
            raise ValueError("Signal must be 1D or 2D")
    
    def _transform_1d(self, signal: np.ndarray) -> np.ndarray:
        """1D forward LWT with multiple levels"""
        result = signal.astype(np.float64)
        all_details = []
        
        for level in range(self.levels):
            if len(result) < 2:
                break
            approx, detail = self._forward_level(result)
            all_details.append(detail)
            result = approx
        
        # Concatenate: [final_approx, detail_n, detail_n-1, ..., detail_1]
        coefficients = [result] + all_details[::-1]
        return np.concatenate(coefficients)
    
    def _transform_2d(self, image: np.ndarray) -> np.ndarray:
        """2D forward LWT (row then column)"""
        result = image.astype(np.float64)
        
        # Transform rows
        for i in range(result.shape[0]):
            result[i] = self._transform_1d(result[i])
        
        # Transform columns
        for j in range(result.shape[1]):
            result[:, j] = self._transform_1d(result[:, j])
        
        return result
    
    def inverse_transform(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Inverse LWT
        
        Args:
            coefficients: Wavelet coefficients
            
        Returns:
            Reconstructed signal
        """
        if len(coefficients.shape) == 1:
            return self._inverse_transform_1d(coefficients)
        elif len(coefficients.shape) == 2:
            return self._inverse_transform_2d(coefficients)
        else:
            raise ValueError("Coefficients must be 1D or 2D")
    
    def _inverse_transform_1d(self, coefficients: np.ndarray) -> np.ndarray:
        """1D inverse LWT"""
        # Split coefficients
        n = len(coefficients)
        level_size = n // (self.levels + 1)
        
        approx = coefficients[:level_size]
        details = []
        
        for i in range(self.levels):
            start = level_size + i * level_size
            end = start + level_size
            if end <= n:
                details.append(coefficients[start:end])
        
        details = details[::-1]  # Reverse order
        
        # Reconstruct
        result = approx
        for detail in details:
            if len(detail) == len(result):
                result = self._inverse_level(result, detail)
            elif len(detail) == len(result) - 1:
                # Handle odd length
                result = self._inverse_level(result[:len(detail)*2], detail)
        
        return result
    
    def _inverse_transform_2d(self, coefficients: np.ndarray) -> np.ndarray:
        """2D inverse LWT"""
        result = coefficients.copy()
        
        # Inverse transform columns
        for j in range(result.shape[1]):
            result[:, j] = self._inverse_transform_1d(result[:, j])
        
        # Inverse transform rows
        for i in range(result.shape[0]):
            result[i] = self._inverse_transform_1d(result[i])
        
        return result


# Example usage
if __name__ == '__main__':
    # 1D signal
    signal = np.sin(np.linspace(0, 10, 1024))
    
    # Initialize
    lwt = LiftingWaveletTransform(wavelet='haar', levels=3)
    
    # Forward transform
    coefficients = lwt.transform(signal)
    print(f"Original shape: {signal.shape}")
    print(f"Coefficients shape: {coefficients.shape}")
    
    # Inverse transform
    reconstructed = lwt.inverse_transform(coefficients)
    
    # Verify reconstruction
    error = np.max(np.abs(signal[:len(reconstructed)] - reconstructed))
    print(f"Max reconstruction error: {error:.2e}")
    
    # 2D image
    image = np.random.rand(256, 256)
    coefficients_2d = lwt.transform(image)
    reconstructed_2d = lwt.inverse_transform(coefficients_2d)
    error_2d = np.max(np.abs(image[:reconstructed_2d.shape[0], :reconstructed_2d.shape[1]] - reconstructed_2d))
    print(f"2D reconstruction error: {error_2d:.2e}")
